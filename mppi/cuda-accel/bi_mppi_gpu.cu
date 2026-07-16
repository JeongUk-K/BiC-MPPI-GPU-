#include "bi_mppi_gpu.cuh"
#include "gpu_kmeans.cuh"
#include "mppi_gpu.cuh"   // rollout_kernel 접근을 위해 포함
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>


// Local rollout kernel (ODR-safe: unique name per TU)
DEFINE_FORWARD_ROLLOUT_KERNEL(bi_rollout_kernel)

// backward_rollout_kernel과 guide_rollout_kernel만 정의 (forward는 매크로 인스턴스화 사용)


// ── Backward rollout kernel ─────────────────────────────────────
__global__ void backward_rollout_kernel(
    const double* __restrict__ d_Ub0,
    double* d_Ubi,
    const double* __restrict__ d_noise,
    const double* __restrict__ d_sigma,
    const double* __restrict__ d_x_init,
    const double* __restrict__ d_x_target,
    double* d_costs, double* d_Di,
    bool with_map, const double* d_map,
    int max_row, int max_col, double res,
    const double* d_circles, int n_circ,
    const double* d_rects,   int n_rect,
    int N, int dim_u, int dim_x, int T, double dt, double gamma_u,
    int model_type)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double* Ui_i = d_Ubi + i * (dim_u * T);

    for (int d = 0; d < dim_u; ++d)
        for (int t = 0; t < T; ++t) {
            double raw = d_Ub0[d*T+t] + d_sigma[d]*d_noise[i*(dim_u*T)+d*T+t];
            Ui_i[d*T+t] = raw;
        }
    legacy_cuda_project_control(Ui_i, dim_u, T, model_type);

    double* Di_i = d_Di + i * dim_u;
    for (int d = 0; d < dim_u; ++d) {
        double acc = 0.0;
        for (int t = 0; t < T; ++t) acc += Ui_i[d*T+t] - d_Ub0[d*T+t];
        Di_i[d] = acc / T;
    }

    double x[GPU_MAX_DIM_X], xn[GPU_MAX_DIM_X], xd_[GPU_MAX_DIM_X];
    for (int d = 0; d < dim_x; ++d) x[d] = d_x_target[d];

    double cost = 0.0;
    bool hit = false;
    if (legacy_cuda_collision_grid(x, with_map, d_map, max_row, max_col, res,
                                   d_circles, n_circ, d_rects, n_rect)) {
        hit = true; cost = 1e8;
    }

    for (int j = T-1; j >= 0; --j) {
        if (hit) break;
        double u_local[GPU_MAX_DIM_U];
        for (int _d = 0; _d < dim_u; ++_d) {
            if (j == T-1) u_local[_d] = Ui_i[_d*T+j];
            else          u_local[_d] = Ui_i[_d*T+j+1];
        }
        legacy_cuda_dynamics(x, u_local, xd_, dim_x, dim_u, model_type);
        for (int d = 0; d < dim_x; ++d) xn[d] = x[d] - dt * xd_[d];
        for (int d = 0; d < dim_x; ++d) x[d] = xn[d];
        cost += legacy_cuda_terminal_cost(x, d_x_init, dim_x, model_type);
        if (legacy_cuda_collision_grid(x, with_map, d_map, max_row, max_col, res,
                                       d_circles, n_circ, d_rects, n_rect)) {
            hit = true; cost = 1e8;
        }
    }
    if (!hit) {
        cost += legacy_cuda_terminal_cost(x, d_x_init, dim_x, model_type);
    }
    d_costs[i] = cost;
}

// ── Guide rollout kernel ────────────────────────────────────────
__global__ void guide_rollout_kernel(
    const double* __restrict__ d_Ur0,
    double* d_Uri,
    const double* __restrict__ d_noise,
    const double* __restrict__ d_sigma,
    const double* __restrict__ d_x_init,
    const double* __restrict__ d_x_target,
    const double* __restrict__ d_Xref,   // dim_x x (Tr+1)
    double* d_costs,
    bool with_map, const double* d_map,
    int max_row, int max_col, double res,
    const double* d_circles, int n_circ,
    const double* d_rects,   int n_rect,
    int N, int dim_u, int dim_x, int Tr, double dt, double gamma_u,
    int model_type)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double* Ui_i = d_Uri + i * (dim_u * Tr);
    for (int d = 0; d < dim_u; ++d)
        for (int t = 0; t < Tr; ++t) {
            Ui_i[d*Tr+t] = d_Ur0[d*Tr+t] +
                            d_sigma[d]*d_noise[i*(dim_u*Tr)+d*Tr+t];
        }
    legacy_cuda_project_control(Ui_i, dim_u, Tr, model_type);

    double x[GPU_MAX_DIM_X], xn[GPU_MAX_DIM_X], xd_[GPU_MAX_DIM_X];
    for (int d = 0; d < dim_x; ++d) x[d] = d_x_init[d];

    double cost = 0.0;
    bool hit = false;
    double guide_cost = 0.0;
    if (legacy_cuda_collision_grid(x, with_map, d_map, max_row, max_col, res,
                                   d_circles, n_circ, d_rects, n_rect)) {
        hit = true; cost = 1e8;
    }

    for (int t = 0; t < Tr; ++t) {
        if (hit) break;
        cost += legacy_cuda_terminal_cost(x, d_x_target, dim_x, model_type);
        double gc = 0.0;
        for (int d = 0; d < dim_x; ++d) {
            double diff = x[d] - d_Xref[d*(Tr+1)+t];
            gc += diff*diff;
        }
        guide_cost += sqrt(gc);
        double u_local[GPU_MAX_DIM_U];
        for (int _d = 0; _d < dim_u; ++_d) u_local[_d] = Ui_i[_d*Tr+t];
        legacy_cuda_dynamics(x, u_local, xd_, dim_x, dim_u, model_type);
        for (int d = 0; d < dim_x; ++d) xn[d] = x[d] + dt*xd_[d];

        for (int d = 0; d < dim_x; ++d) x[d] = xn[d];
        if (legacy_cuda_collision_grid(x, with_map, d_map, max_row, max_col,
                                       res, d_circles, n_circ, d_rects,
                                       n_rect)) {
            hit = true; cost = 1e8;
        }
    }
    if (!hit) {
        double gc = 0.0;
        for (int d = 0; d < dim_x; ++d) {
            double diff = x[d] - d_Xref[d*(Tr+1)+Tr];
            gc += diff*diff;
        }
        guide_cost += sqrt(gc);
        cost = legacy_cuda_terminal_cost(x, d_x_target, dim_x, model_type);
        cost += guide_cost;
    }
    d_costs[i] = cost;
}

// ── Helper: safe malloc ─────────────────────────────────────────
static void safe_cuda_malloc(double** ptr, size_t sz) {
    if (*ptr) { cudaFree(*ptr); *ptr = nullptr; }
    if (sz > 0) CUDA_CHECK(cudaMalloc(ptr, sz));
}

// ── BiMPPI_GPU methods ──────────────────────────────────────────
BiMPPI_GPU::~BiMPPI_GPU() {
    freeForward(); freeBackward(); freeGuide(); freeCommon();
    curandDestroyGenerator(curand_gen);
}

void BiMPPI_GPU::freeForward() {
    safe_cuda_malloc(&d_Uf0, 0); safe_cuda_malloc(&d_Ufi, 0);
    safe_cuda_malloc(&d_noise_f, 0); safe_cuda_malloc(&d_costs_f, 0);
    safe_cuda_malloc(&d_Uf_out, 0); safe_cuda_malloc(&d_Di_f, 0);
}
void BiMPPI_GPU::freeBackward() {
    safe_cuda_malloc(&d_Ub0, 0); safe_cuda_malloc(&d_Ubi, 0);
    safe_cuda_malloc(&d_noise_b, 0); safe_cuda_malloc(&d_costs_b, 0);
    safe_cuda_malloc(&d_Ub_out, 0); safe_cuda_malloc(&d_Di_b, 0);
}
void BiMPPI_GPU::freeGuide() {
    safe_cuda_malloc(&d_Xref, 0);
    safe_cuda_malloc(&d_Ur0, 0); safe_cuda_malloc(&d_Uri, 0);
    safe_cuda_malloc(&d_noise_r, 0); safe_cuda_malloc(&d_costs_r, 0);
    safe_cuda_malloc(&d_Ur_out, 0);
    alloc_Tr_guide = 0;
}
void BiMPPI_GPU::freeCommon() {
    safe_cuda_malloc(&d_x_init, 0); safe_cuda_malloc(&d_x_target, 0);
    safe_cuda_malloc(&d_sigma, 0);
    safe_cuda_malloc(&d_map, 0); safe_cuda_malloc(&d_circles, 0);
    safe_cuda_malloc(&d_rects, 0);
}

void BiMPPI_GPU::allocForward() {
    safe_cuda_malloc(&d_Uf0,    (size_t)dim_u*Tf*sizeof(double));
    safe_cuda_malloc(&d_Ufi,    (size_t)Nf*dim_u*Tf*sizeof(double));
    safe_cuda_malloc(&d_noise_f,(size_t)Nf*dim_u*Tf*sizeof(double));
    safe_cuda_malloc(&d_costs_f,(size_t)Nf*sizeof(double));
    safe_cuda_malloc(&d_Uf_out, (size_t)dim_u*Tf*sizeof(double));
    safe_cuda_malloc(&d_Di_f,   (size_t)Nf*dim_u*sizeof(double));
}
void BiMPPI_GPU::allocBackward() {
    safe_cuda_malloc(&d_Ub0,    (size_t)dim_u*Tb*sizeof(double));
    safe_cuda_malloc(&d_Ubi,    (size_t)Nb*dim_u*Tb*sizeof(double));
    safe_cuda_malloc(&d_noise_b,(size_t)Nb*dim_u*Tb*sizeof(double));
    safe_cuda_malloc(&d_costs_b,(size_t)Nb*sizeof(double));
    safe_cuda_malloc(&d_Ub_out, (size_t)dim_u*Tb*sizeof(double));
    safe_cuda_malloc(&d_Di_b,   (size_t)Nb*dim_u*sizeof(double));
}

void BiMPPI_GPU::allocGuide() {
    allocGuideFor(Tf + Tb);
}

void BiMPPI_GPU::allocGuideFor(int Tr) {
    if (Tr <= alloc_Tr_guide && d_Xref && d_Ur0 && d_Uri && d_noise_r &&
        d_costs_r && d_Ur_out) {
        return;
    }
    safe_cuda_malloc(&d_Xref, (size_t)dim_x*(Tr+1)*sizeof(double));
    safe_cuda_malloc(&d_Ur0,  (size_t)dim_u*Tr*sizeof(double));
    safe_cuda_malloc(&d_Uri,  (size_t)Nr*dim_u*Tr*sizeof(double));
    size_t ncnt = (size_t)Nr*dim_u*Tr; if (ncnt % 2) ncnt++;
    safe_cuda_malloc(&d_noise_r, ncnt*sizeof(double));
    safe_cuda_malloc(&d_costs_r, (size_t)Nr*sizeof(double));
    safe_cuda_malloc(&d_Ur_out,  (size_t)dim_u*Tr*sizeof(double));
    alloc_Tr_guide = Tr;
}

void BiMPPI_GPU::init(BiMPPIParam p) {
    dt = p.dt; Tf = p.Tf; Tb = p.Tb;
    Nf = p.Nf; Nb = p.Nb; Nr = p.Nr;
    gamma_u = p.gamma_u;
    x_init = p.x_init; x_target = p.x_target;
    deviation_mu = p.deviation_mu; cost_mu = p.cost_mu; epsilon = p.epsilon;
    minpts = p.minpts; psi = p.psi;
    kmeans_clusters = p.kmeans_clusters;
    kmeans_max_iterations = p.kmeans_max_iterations;
    kmeans_threshold = p.kmeans_threshold;

    sigma_diag.resize(dim_u);
    for (int d = 0; d < dim_u; ++d) sigma_diag[d] = p.sigma_u(d,d);

    full_cluster_f.resize(Nf); std::iota(full_cluster_f.begin(), full_cluster_f.end(), 0);
    full_cluster_b.resize(Nb); std::iota(full_cluster_b.begin(), full_cluster_b.end(), 0);

    u0 = Eigen::VectorXd::Zero(dim_u);
    dummy_u = Eigen::VectorXd::Zero(dim_u);

    CUDA_CHECK(cudaMalloc(&d_x_init,   dim_x*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x_target, dim_x*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sigma,    dim_u*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_sigma, sigma_diag.data(), dim_u*sizeof(double), cudaMemcpyHostToDevice));

    allocForward(); allocBackward(); allocGuide();
    alloc_Nf=Nf; alloc_Nb=Nb; alloc_Tf=Tf; alloc_Tb=Tb;
}

void BiMPPI_GPU::setCollisionChecker(CollisionChecker* cc) {
    collision_checker = cc;
    uploadCollisionData();
}

void BiMPPI_GPU::setConnectionMetric(ConnectionMetric metric) {
    connection_metric = metric;
}

void BiMPPI_GPU::setSE2ConnectionWeights(double xy_weight,
                                         double theta_weight) {
    if (xy_weight < 0.0 || theta_weight < 0.0 ||
        (xy_weight == 0.0 && theta_weight == 0.0)) {
        throw std::invalid_argument(
            "SE(2) connection weights must be non-negative and not both zero.");
    }
    se2_connection_xy_weight = xy_weight;
    se2_connection_theta_weight = theta_weight;
}

void BiMPPI_GPU::uploadCollisionData() {
    map_max_row = (int)collision_checker->map.size();
    map_max_col = map_max_row>0?(int)collision_checker->map[0].size():0;
    map_resolution = collision_checker->resolution;
    with_map = collision_checker->with_map;

    if (with_map && map_max_row>0) {
        size_t sz = (size_t)map_max_row*map_max_col*sizeof(double);
        safe_cuda_malloc(&d_map, sz);
        std::vector<double> flat(map_max_row*map_max_col);
        for (int r=0;r<map_max_row;++r)
            for (int c=0;c<map_max_col;++c)
                flat[r*map_max_col+c]=collision_checker->map[r][c];
        CUDA_CHECK(cudaMemcpy(d_map,flat.data(),sz,cudaMemcpyHostToDevice));
    }
    n_circles=(int)collision_checker->circles.size();
    if (n_circles>0) {
        size_t sz=n_circles*4*sizeof(double);
        safe_cuda_malloc(&d_circles,sz);
        std::vector<double> buf(n_circles*4);
        for (int i=0;i<n_circles;++i) for(int j=0;j<4;++j) buf[i*4+j]=collision_checker->circles[i][j];
        CUDA_CHECK(cudaMemcpy(d_circles,buf.data(),sz,cudaMemcpyHostToDevice));
    }
    n_rects=(int)collision_checker->rectangles.size();
    if (n_rects>0) {
        size_t sz=n_rects*4*sizeof(double);
        safe_cuda_malloc(&d_rects,sz);
        std::vector<double> buf(n_rects*4);
        for (int i=0;i<n_rects;++i) for(int j=0;j<4;++j) buf[i*4+j]=collision_checker->rectangles[i][j];
        CUDA_CHECK(cudaMemcpy(d_rects,buf.data(),sz,cudaMemcpyHostToDevice));
    }
}

// ── Helper: Eigen → flat row-major ────────────────────────────────
static std::vector<double> eigen_to_flat(const Eigen::MatrixXd& M, int rows, int cols) {
    std::vector<double> v(rows*cols);
    for(int r=0;r<rows;++r) for(int c=0;c<cols;++c) v[r*cols+c]=M(r,c);
    return v;
}
static Eigen::MatrixXd flat_Ui_to_eigen(const std::vector<double>& f, int N, int du, int T) {
    Eigen::MatrixXd M(N*du, T);
    for(int i=0;i<N;++i) for(int d=0;d<du;++d) for(int t=0;t<T;++t)
        M(i*du+d,t)=f[i*(du*T)+d*T+t];
    return M;
}

void BiMPPI_GPU::appendVisRolloutSamples(const Eigen::MatrixXd &Ui_cpu,
                                         int N_samples, int T_steps,
                                         bool backward) {
    if (!vis_logger || !vis_logger->enabled || N_samples <= 0 ||
        T_steps <= 0 || Ui_cpu.rows() < N_samples * dim_u) {
        return;
    }

    if (vis_rollout_samples.size() >= kMaxSavedBiRollouts) {
        return;
    }

    const std::size_t remaining = kMaxSavedBiRollouts - vis_rollout_samples.size();
    const int wanted =
        std::max(1, std::min(vis_rollout_samples_per_call,
                             static_cast<int>(remaining)));
    const int stride = std::max(1, static_cast<int>(std::ceil(
                                       static_cast<double>(N_samples) /
                                       static_cast<double>(wanted))));

    for (int i = 0; i < N_samples &&
                    vis_rollout_samples.size() < kMaxSavedBiRollouts;
         i += stride) {
        Eigen::MatrixXd X = Eigen::MatrixXd::Zero(dim_x, T_steps + 1);
        if (!backward) {
            X.col(0) = x_init;
            for (int t = 0; t < T_steps; ++t) {
                const Eigen::VectorXd u =
                    Ui_cpu.block(i * dim_u, t, dim_u, 1);
                X.col(t + 1) = X.col(t) + (double)dt * f(X.col(t), u);
            }
        } else {
            X.col(T_steps) = x_target;
            for (int t = T_steps - 1; t >= 0; --t) {
                const int u_col = (t == T_steps - 1) ? t : t + 1;
                const Eigen::VectorXd u =
                    Ui_cpu.block(i * dim_u, u_col, dim_u, 1);
                X.col(t) = X.col(t + 1) - (double)dt * f(X.col(t + 1), u);
            }
        }
        vis_rollout_samples.push_back(std::move(X));
    }
}

// ── Raw rollout helpers for ablation variants ───────────────────
void BiMPPI_GPU::forwardRawRollout(Eigen::VectorXd &costs,
                                   Eigen::MatrixXd &Ui_cpu) {
    auto t0=std::chrono::high_resolution_clock::now();
    auto ff=eigen_to_flat(U_f0,dim_u,Tf);
    CUDA_CHECK(cudaMemcpy(d_Uf0,ff.data(),dim_u*Tf*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_init,  x_init.data(),  dim_x*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_target,x_target.data(),dim_x*sizeof(double),cudaMemcpyHostToDevice));
    size_t nc=(size_t)Nf*dim_u*Tf; if(nc%2)nc++;
    CURAND_CHECK(curandGenerateNormalDouble(curand_gen,d_noise_f,nc,0.0,1.0));
    int B=256,G=(Nf+B-1)/B;
    bi_rollout_kernel<<<G,B>>>(d_Uf0,d_Ufi,d_noise_f,d_sigma,d_x_init,d_x_target,
        d_costs_f,d_Di_f,with_map,d_map,map_max_row,map_max_col,map_resolution,
        d_circles,n_circles,d_rects,n_rects,Nf,dim_u,dim_x,Tf,(double)dt,gamma_u,model_type,true);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    auto t1=std::chrono::high_resolution_clock::now();
    elapsed_rollout+=std::chrono::duration<double>(t1-t0).count();

    std::vector<double> hc(Nf),hUi((size_t)Nf*dim_u*Tf);
    CUDA_CHECK(cudaMemcpy(hc.data(), d_costs_f,Nf*sizeof(double),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hUi.data(),d_Ufi,(size_t)Nf*dim_u*Tf*sizeof(double),cudaMemcpyDeviceToHost));
    costs=Eigen::Map<Eigen::VectorXd>(hc.data(),Nf);
    Ui_cpu=flat_Ui_to_eigen(hUi,Nf,dim_u,Tf);
}

void BiMPPI_GPU::backwardRawRollout(Eigen::VectorXd &costs,
                                    Eigen::MatrixXd &Ui_cpu) {
    auto t0=std::chrono::high_resolution_clock::now();
    auto bf=eigen_to_flat(U_b0,dim_u,Tb);
    CUDA_CHECK(cudaMemcpy(d_x_init,  x_init.data(),  dim_x*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_target,x_target.data(),dim_x*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Ub0,bf.data(),dim_u*Tb*sizeof(double),cudaMemcpyHostToDevice));
    size_t nc=(size_t)Nb*dim_u*Tb; if(nc%2)nc++;
    CURAND_CHECK(curandGenerateNormalDouble(curand_gen,d_noise_b,nc,0.0,1.0));
    int B=256,G=(Nb+B-1)/B;
    backward_rollout_kernel<<<G,B>>>(d_Ub0,d_Ubi,d_noise_b,d_sigma,d_x_init,d_x_target,
        d_costs_b,d_Di_b,with_map,d_map,map_max_row,map_max_col,map_resolution,
        d_circles,n_circles,d_rects,n_rects,Nb,dim_u,dim_x,Tb,(double)dt,gamma_u,model_type);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    auto t1=std::chrono::high_resolution_clock::now();
    elapsed_rollout+=std::chrono::duration<double>(t1-t0).count();

    std::vector<double> hc(Nb),hUi((size_t)Nb*dim_u*Tb);
    CUDA_CHECK(cudaMemcpy(hc.data(), d_costs_b,Nb*sizeof(double),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hUi.data(),d_Ubi,(size_t)Nb*dim_u*Tb*sizeof(double),cudaMemcpyDeviceToHost));
    costs=Eigen::Map<Eigen::VectorXd>(hc.data(),Nb);
    Ui_cpu=flat_Ui_to_eigen(hUi,Nb,dim_u,Tb);
}

// ── fastsc K-means (GPU) ──────────────────────────────────────────
void BiMPPI_GPU::kmeansCluster(std::vector<std::vector<int>>& clusters,
                         const Eigen::MatrixXd& feature_source,
                         const Eigen::VectorXd& costs, int Ns) {
    clusters.clear();

    // feature_source supports two layouts:
    //   1) dim_feature x Ns       : use as a precomputed feature matrix.
    //   2) (Ns * dim_u) x T_steps : sampled control sequences Ui_cpu.
    //      In this case, build a 3-block input feature internally:
    //        D_i = [mean_{0:T/3} W_u u_i,
    //               mean_{T/3:2T/3} W_u u_i,
    //               mean_{2T/3:T} W_u u_i].
    //      This is distance-equivalent to using W_u (u_i - u_0), because
    //      the same warm-start u_0 is subtracted from every sample and
    //      therefore cancels in pairwise DBSCAN distances.
    constexpr int kNumBlocks = 3;
    Eigen::MatrixXd block_feature;
    const Eigen::MatrixXd* feature_ptr = &feature_source;

    const bool source_is_control_sequence =
        (feature_source.rows() == Ns * dim_u && feature_source.cols() > 0);

    if (source_is_control_sequence) {
        const int T_steps = static_cast<int>(feature_source.cols());
        block_feature = Eigen::MatrixXd::Zero(kNumBlocks * dim_u, Ns);

        for (int i = 0; i < Ns; ++i) {
            for (int b = 0; b < kNumBlocks; ++b) {
                const int t0 = (b * T_steps) / kNumBlocks;
                const int t1 = ((b + 1) * T_steps) / kNumBlocks;
                const int len = std::max(1, t1 - t0);

                for (int d = 0; d < dim_u; ++d) {
                    double acc = 0.0;
                    for (int t = t0; t < t1; ++t) {
                        acc += feature_source(i * dim_u + d, t);
                    }

                    // W_u.  Keep identity scaling to preserve the original
                    // epsilon unit as much as possible.  To normalize by the
                    // sampling standard deviation, replace scale with:
                    //   std::max(1e-12, std::abs(sigma_diag[d]))
                    const double scale = 1.0;
                    block_feature(b * dim_u + d, i) = (acc / static_cast<double>(len)) / scale;
                }
            }
        }
        feature_ptr = &block_feature;
    }

    const Eigen::MatrixXd& feature = *feature_ptr;

    runGPUKMeans(clusters, feature, costs,
                 {kmeans_clusters, kmeans_max_iterations,
                  kmeans_threshold});
    return;

#if 0 // Former CPU DBSCAN retained temporarily for algorithm comparison.

    std::vector<std::vector<int>> upper_neighbors(Ns);
    std::vector<char> valid(Ns, false);
    std::vector<int> valid_indices;
    valid_indices.reserve(std::max(0, Ns));

    constexpr double J_col = 1e8;
    int best_idx = 0;
    double best_cost = (Ns > 0) ? costs(0) : 0.0;
    for (int i = 0; i < Ns; ++i) {
        if (costs(i) < best_cost) {
            best_cost = costs(i);
            best_idx = i;
        }
        valid[i] = (costs(i) < J_col);
        if (valid[i]) valid_indices.push_back(i);
    }

    // If every sample is collision-penalized, keep the least-cost sample as
    // the fallback branch instead of averaging all invalid samples.
    if (valid_indices.empty()) {
        if (Ns > 0) clusters.push_back(std::vector<int>{best_idx});
        return;
    }

    const bool use_squared_distance = (deviation_mu > 0.0 && epsilon > 0.0);
    const double eps_scaled = use_squared_distance ? epsilon / deviation_mu : 0.0;
    const double eps_scaled_sq = eps_scaled * eps_scaled;

#pragma omp parallel for schedule(dynamic,16)
    for (int i = 0; i < Ns; ++i) {
        if (!valid[i]) continue;
        std::vector<int>& neighbors = upper_neighbors[i];
        for (int j = i + 1; j < Ns; ++j) {
            if (!valid[j]) continue;
            double dist_sq = 0.0;
            for (int d = 0; d < feature.rows(); ++d) {
                const double diff = feature(d, i) - feature(d, j);
                dist_sq += diff * diff;
            }
            const bool is_neighbor = use_squared_distance
                ? (dist_sq < eps_scaled_sq)
                : (deviation_mu * std::sqrt(dist_sq) < epsilon);
            if (is_neighbor) neighbors.push_back(j);
        }
    }

    std::vector<size_t> degrees(Ns, 0);
    for (int i = 0; i < Ns; ++i) {
        degrees[i] += upper_neighbors[i].size();
        for (int j : upper_neighbors[i]) ++degrees[j];
    }

    std::vector<std::vector<int>> tree(Ns);
    for (int i = 0; i < Ns; ++i) tree[i].reserve(degrees[i]);
    for (int i = 0; i < Ns; ++i) {
        for (int j : upper_neighbors[i]) {
            tree[i].push_back(j);
            tree[j].push_back(i);
        }
    }

    std::vector<char> core(Ns, false);
    for (int i = 0; i < Ns; ++i) {
        if ((int)tree[i].size() > minpts) core[i] = true;
    }

    std::vector<char> vis(Ns, false);
    for (int i = 0; i < Ns; ++i) {
        if (!core[i] || vis[i]) continue;
        std::deque<int> q;
        std::vector<int> cl;
        q.push_back(i);
        cl.push_back(i);
        vis[i] = true;
        while (!q.empty()) {
            const int n = q.front();
            q.pop_front();
            for (int nb : tree[n]) {
                if (vis[nb]) continue;
                vis[nb] = true;
                cl.push_back(nb);
                if (core[nb]) q.push_back(nb);
            }
        }
        clusters.push_back(cl);
    }

    // DBSCAN may classify all valid samples as noise.  In that case, retain a
    // single fallback cluster containing only valid samples.
    if (clusters.empty()) {
        clusters.push_back(std::move(valid_indices));
    }
#endif
}

// ── calculateU (CPU) ──────────────────────────────────────────────
void BiMPPI_GPU::calculateU(Eigen::MatrixXd& Uout,
                             const std::vector<std::vector<int>>& clusters,
                             const Eigen::VectorXd& costs,
                             const Eigen::MatrixXd& Ui_cpu, int T_steps) {
    int nc=(int)clusters.size();
    Uout=Eigen::MatrixXd::Zero(nc*dim_u, T_steps);
#pragma omp parallel for
    for(int idx=0;idx<nc;++idx){
        int pts=(int)clusters[idx].size();
        double mc=std::numeric_limits<double>::max();
        for(int k:clusters[idx]) mc=std::min(mc,costs(k));
        double tw=0.0; std::vector<double> wts(pts);
        for(int i=0;i<pts;++i){ wts[i]=std::exp(-gamma_u*(costs(clusters[idx][i])-mc)); tw+=wts[i]; }
        for(int i=0;i<pts;++i)
            Uout.middleRows(idx*dim_u,dim_u)+=(wts[i]/tw)*Ui_cpu.middleRows(clusters[idx][i]*dim_u,dim_u);
        Eigen::Ref<Eigen::MatrixXd> slice = Uout.middleRows(idx*dim_u, dim_u);
        h(slice);
    }
}

// ── forwardRollout ────────────────────────────────────────────────
void BiMPPI_GPU::forwardRollout() {
    auto t0=std::chrono::high_resolution_clock::now();
    auto ff=eigen_to_flat(U_f0,dim_u,Tf);
    CUDA_CHECK(cudaMemcpy(d_Uf0,ff.data(),dim_u*Tf*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_init,  x_init.data(),  dim_x*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_target,x_target.data(),dim_x*sizeof(double),cudaMemcpyHostToDevice));
    size_t nc=(size_t)Nf*dim_u*Tf; if(nc%2)nc++;
    CURAND_CHECK(curandGenerateNormalDouble(curand_gen,d_noise_f,nc,0.0,1.0));
    int B=256,G=(Nf+B-1)/B;
    bi_rollout_kernel<<<G,B>>>(d_Uf0,d_Ufi,d_noise_f,d_sigma,d_x_init,d_x_target,
        d_costs_f,d_Di_f,with_map,d_map,map_max_row,map_max_col,map_resolution,
        d_circles,n_circles,d_rects,n_rects,Nf,dim_u,dim_x,Tf,(double)dt,gamma_u,model_type,true);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    auto t1=std::chrono::high_resolution_clock::now();
    elapsed_rollout+=std::chrono::duration<double>(t1-t0).count();

    std::vector<double> hc(Nf), hUi((size_t)Nf * dim_u * Tf);
    CUDA_CHECK(cudaMemcpy(hc.data(), d_costs_f, Nf * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hUi.data(), d_Ufi, (size_t)Nf * dim_u * Tf * sizeof(double), cudaMemcpyDeviceToHost));
    Eigen::VectorXd costs_f = Eigen::Map<Eigen::VectorXd>(hc.data(), Nf);
    Eigen::MatrixXd Ui_f = flat_Ui_to_eigen(hUi, Nf, dim_u, Tf);
    appendVisRolloutSamples(Ui_f, Nf, Tf, false);
    clusters_f.clear();
    kmeansCluster(clusters_f, Ui_f, costs_f, Nf);
    if(clusters_f.empty()) clusters_f.push_back(full_cluster_f);
    calculateU(Uf,clusters_f,costs_f,Ui_f,Tf);
    Xf.resize(clusters_f.size()*dim_x,Tf+1);
    for(int ci=0;ci<(int)clusters_f.size();++ci){
        Xf.block(ci*dim_x,0,dim_x,1)=x_init;
        for(int t=0;t<Tf;++t){
            Xf.block(ci*dim_x,t+1,dim_x,1)=Xf.block(ci*dim_x,t,dim_x,1)+
                (double)dt*f(Xf.block(ci*dim_x,t,dim_x,1), Uf.block(ci*dim_u,t,dim_u,1));
        }
    }
    elapsed_clustering+=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t1).count();
}

// ── backwardRollout ───────────────────────────────────────────────
void BiMPPI_GPU::backwardRollout() {
    auto t0=std::chrono::high_resolution_clock::now();
    auto bf=eigen_to_flat(U_b0,dim_u,Tb);
    CUDA_CHECK(cudaMemcpy(d_x_init,  x_init.data(),  dim_x*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_target,x_target.data(),dim_x*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Ub0,bf.data(),dim_u*Tb*sizeof(double),cudaMemcpyHostToDevice));
    size_t nc=(size_t)Nb*dim_u*Tb; if(nc%2)nc++;
    CURAND_CHECK(curandGenerateNormalDouble(curand_gen,d_noise_b,nc,0.0,1.0));
    int B=256,G=(Nb+B-1)/B;
    backward_rollout_kernel<<<G,B>>>(d_Ub0,d_Ubi,d_noise_b,d_sigma,d_x_init,d_x_target,
        d_costs_b,d_Di_b,with_map,d_map,map_max_row,map_max_col,map_resolution,
        d_circles,n_circles,d_rects,n_rects,Nb,dim_u,dim_x,Tb,(double)dt,gamma_u,model_type);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    auto t1=std::chrono::high_resolution_clock::now();
    elapsed_rollout+=std::chrono::duration<double>(t1-t0).count();

    std::vector<double> hc(Nb), hUi((size_t)Nb * dim_u * Tb);
    CUDA_CHECK(cudaMemcpy(hc.data(), d_costs_b, Nb * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hUi.data(), d_Ubi, (size_t)Nb * dim_u * Tb * sizeof(double), cudaMemcpyDeviceToHost));
    Eigen::VectorXd costs_b = Eigen::Map<Eigen::VectorXd>(hc.data(), Nb);
    Eigen::MatrixXd Ui_b = flat_Ui_to_eigen(hUi, Nb, dim_u, Tb);
    appendVisRolloutSamples(Ui_b, Nb, Tb, true);
    clusters_b.clear();
    kmeansCluster(clusters_b, Ui_b, costs_b, Nb);
    if(clusters_b.empty()) clusters_b.push_back(full_cluster_b);
    calculateU(Ub,clusters_b,costs_b,Ui_b,Tb);
    Xb.resize(clusters_b.size()*dim_x,Tb+1);
    for(int ci=0;ci<(int)clusters_b.size();++ci){
        Xb.block(ci*dim_x,Tb,dim_x,1)=x_target;
        for(int t=Tb-1;t>=0;--t){
            Eigen::VectorXd u_val(dim_u);
            if(t==Tb-1){ u_val=Ub.block(ci*dim_u,t,dim_u,1); }
            else        { u_val=Ub.block(ci*dim_u,t+1,dim_u,1); }
            Xb.block(ci*dim_x,t,dim_x,1)=Xb.block(ci*dim_x,t+1,dim_x,1)-(double)dt*f(Xb.block(ci*dim_x,t+1,dim_x,1), u_val);
        }
    }
    elapsed_clustering+=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t1).count();
}

// ── selectConnection ──────────────────────────────────────────────
double BiMPPI_GPU::connectionMetricDistance(
    const Eigen::Ref<const Eigen::VectorXd> &xf,
    const Eigen::Ref<const Eigen::VectorXd> &xb) const {
    if (connection_metric == ConnectionMetric::SE2 && dim_x >= 3) {
        const double dx = xf(0) - xb(0);
        const double dy = xf(1) - xb(1);
        const double dtheta = std::atan2(std::sin(xf(2) - xb(2)),
                                         std::cos(xf(2) - xb(2)));
        return std::sqrt(se2_connection_xy_weight * (dx * dx + dy * dy) +
                         se2_connection_theta_weight * (dtheta * dtheta));
    }

    return (xf - xb).norm();
}

void BiMPPI_GPU::selectConnection() {
    joints.clear();
    for(int cf=0;cf<(int)clusters_f.size();++cf){
        double mn=std::numeric_limits<double>::max(); int cb=0,df_=0,db_=0;
        for(int cb_=0;cb_<(int)clusters_b.size();++cb_)
            for(int df__=0;df__<=Tf;++df__)
                for(int db__=0;db__<=Tb;++db__){
                    double n=(Xf.block(cf*dim_x,df__,dim_x,1)-Xb.block(cb_*dim_x,db__,dim_x,1)).norm();
                    if (connection_metric == ConnectionMetric::SE2) {
                        n = connectionMetricDistance(
                            Xf.block(cf * dim_x, df__, dim_x, 1),
                            Xb.block(cb_ * dim_x, db__, dim_x, 1));
                    }
                    if(n<mn){mn=n;cb=cb_;df_=df__;db_=db__;}
                }
        joints.push_back({cf,cb,df_,db_});
    }
}

double BiMPPI_GPU::connectionDistance() const {
    double total = 0.0;
    int count = 0;
    for (const auto& joint : joints) {
        if (joint.size() < 4) continue;
        int cf = joint[0], cb = joint[1], df = joint[2], db = joint[3];
        if (cf < 0 || cb < 0 || df < 0 || db < 0 ||
            cf >= (int)clusters_f.size() || cb >= (int)clusters_b.size() ||
            df > Tf || db > Tb) {
            continue;
        }
        double distance = (Xf.block(cf * dim_x, df, dim_x, 1) -
                           Xb.block(cb * dim_x, db, dim_x, 1))
                              .norm();
        if (connection_metric == ConnectionMetric::SE2) {
            distance = connectionMetricDistance(
                Xf.block(cf * dim_x, df, dim_x, 1),
                Xb.block(cb * dim_x, db, dim_x, 1));
        }
        total += distance;
        ++count;
    }
    if (count == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return total / count;
}

// ── concatenate ───────────────────────────────────────────────────
void BiMPPI_GPU::concatenate() {
    Uc.clear(); Xc.clear();
    for(auto& j:joints){
        int cf=j[0],cb=j[1],df=j[2],db=j[3],len=std::max(Tf,df+(Tb-db));
        Eigen::MatrixXd U=Eigen::MatrixXd::Zero(dim_u,len);
        Eigen::MatrixXd X=Eigen::MatrixXd::Zero(dim_x,len+1);
        if(df==0) {
            X.leftCols(df+1)=Xf.block(cf*dim_x,0,dim_x,df+1);
        } else {
            U.leftCols(df)=Uf.block(cf*dim_u,0,dim_u,df);
            X.leftCols(df+1)=Xf.block(cf*dim_x,0,dim_x,df+1);
        }
        if(db!=Tb){
            U.middleCols(df,Tb-db)=Ub.block(cb*dim_u,db,dim_u,Tb-db);
            X.middleCols(df+1,Tb-db)=Xb.block(cb*dim_x,db+1,dim_x,Tb-db);
        }
        if(df+(Tb-db)<Tf){ U.rightCols(Tf-(df+(Tb-db))).colwise()=dummy_u;
                           X.rightCols(Tf-(df+(Tb-db))).colwise()=x_target; }
        Uc.push_back(U); Xc.push_back(X);
    }
}

// ── guideMPPI ─────────────────────────────────────────────────────
void BiMPPI_GPU::guideMPPI() {
    Ur.clear(); Cr.clear(); Xr.clear();
    for(int r=0;r<(int)joints.size();++r){
        int Tr=Uc[r].cols();
        allocGuideFor(Tr);
        std::vector<double> xrf(dim_x*(Tr+1));
        for(int d=0;d<dim_x;++d) for(int t=0;t<=Tr;++t) xrf[d*(Tr+1)+t]=Xc[r](d,t);
        CUDA_CHECK(cudaMemcpy(d_Xref,xrf.data(),(size_t)dim_x*(Tr+1)*sizeof(double),cudaMemcpyHostToDevice));
        auto r0f=eigen_to_flat(Uc[r],dim_u,Tr);
        CUDA_CHECK(cudaMemcpy(d_Ur0,r0f.data(),(size_t)dim_u*Tr*sizeof(double),cudaMemcpyHostToDevice));
        size_t ncnt=(size_t)Nr*dim_u*Tr; if(ncnt%2)ncnt++;
        CURAND_CHECK(curandGenerateNormalDouble(curand_gen,d_noise_r,ncnt,0.0,1.0));
        int B=256,G=(Nr+B-1)/B;
        guide_rollout_kernel<<<G,B>>>(d_Ur0,d_Uri,d_noise_r,d_sigma,d_x_init,d_x_target,d_Xref,d_costs_r,
            with_map,d_map,map_max_row,map_max_col,map_resolution,d_circles,n_circles,d_rects,n_rects,
            Nr,dim_u,dim_x,Tr,(double)dt,gamma_u,model_type);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> hcr(Nr),hri((size_t)Nr*dim_u*Tr);
        CUDA_CHECK(cudaMemcpy(hcr.data(),d_costs_r,Nr*sizeof(double),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hri.data(),d_Uri,(size_t)Nr*dim_u*Tr*sizeof(double),cudaMemcpyDeviceToHost));
        Eigen::VectorXd costs_r=Eigen::Map<Eigen::VectorXd>(hcr.data(),Nr);
        double mc=costs_r.minCoeff();
        Eigen::VectorXd wts=(-gamma_u*(costs_r.array()-mc)).exp(); wts/=wts.sum();
        Eigen::MatrixXd Ures=Eigen::MatrixXd::Zero(dim_u,Tr);
        for(int i=0;i<Nr;++i) for(int d=0;d<dim_u;++d) for(int t=0;t<Tr;++t)
            Ures(d,t)+=wts(i)*hri[i*(dim_u*Tr)+d*Tr+t];
        h(Ures);
        Eigen::MatrixXd Xi(dim_x,Tr+1); Xi.col(0)=x_init; double cost=0.0;
        for(int t=0;t<Tr;++t){
            Xi.col(t+1)=Xi.col(t)+(double)dt*f(Xi.col(t), Ures.col(t));
            cost+=p(Xi.col(t), x_target);
        }
        cost+=p(Xi.col(Tr), x_target);
        for(int t=0;t<Tr+1;++t){
            if(collision_checker->getCollisionGrid(Xi.col(t))){
                cost=1e8;
                break;
            }
        }
        Ur.push_back(Ures); Cr.push_back(cost); Xr.push_back(Xi);
    }
    double mn=std::numeric_limits<double>::max(); int idx=0;
    for(int r=0;r<(int)joints.size();++r) if(Cr[r]<mn){mn=Cr[r];idx=r;}
    if(mn>=1e7){
        double best_ref_cost=std::numeric_limits<double>::max();
        int best_ref_idx=-1;
        for(int r=0;r<(int)Xc.size();++r){
            bool feasible=true;
            double ref_cost=0.0;
            for(int t=0;t<Xc[r].cols();++t){
                if(collision_checker->getCollisionGrid(Xc[r].col(t))){
                    feasible=false;
                    break;
                }
                ref_cost+=p(Xc[r].col(t), x_target);
            }
            if(feasible && ref_cost<best_ref_cost){
                best_ref_cost=ref_cost;
                best_ref_idx=r;
            }
        }
        if(best_ref_idx>=0){
            Uo=Uc[best_ref_idx]; Xo=Xc[best_ref_idx]; u0=Uo.col(0);
            return;
        }
    }
    Uo=Ur[idx]; Xo=Xr[idx]; u0=Uo.col(0);
}

void BiMPPI_GPU::guideReference(const Eigen::MatrixXd &Uref,
                                const Eigen::MatrixXd &Xref) {
    const int Tr = static_cast<int>(Uref.cols());
    if (Tr <= 0 || Uref.rows() != dim_u || Xref.rows() != dim_x ||
        Xref.cols() != Tr + 1) {
        throw std::runtime_error("Invalid guide reference dimensions");
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    allocGuideFor(Tr);

    std::vector<double> xrf(dim_x * (Tr + 1));
    for (int d = 0; d < dim_x; ++d) {
        for (int t = 0; t <= Tr; ++t) {
            xrf[d * (Tr + 1) + t] = Xref(d, t);
        }
    }
    CUDA_CHECK(cudaMemcpy(d_Xref, xrf.data(),
                          (size_t)dim_x * (Tr + 1) * sizeof(double),
                          cudaMemcpyHostToDevice));

    auto r0f = eigen_to_flat(Uref, dim_u, Tr);
    CUDA_CHECK(cudaMemcpy(d_Ur0, r0f.data(), (size_t)dim_u * Tr * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_init, x_init.data(), dim_x * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_target, x_target.data(), dim_x * sizeof(double),
                          cudaMemcpyHostToDevice));

    size_t ncnt = (size_t)Nr * dim_u * Tr;
    if (ncnt % 2) ncnt++;
    CURAND_CHECK(curandGenerateNormalDouble(curand_gen, d_noise_r, ncnt, 0.0,
                                            1.0));
    int B = 256, G = (Nr + B - 1) / B;
    guide_rollout_kernel<<<G, B>>>(
        d_Ur0, d_Uri, d_noise_r, d_sigma, d_x_init, d_x_target, d_Xref,
        d_costs_r, with_map, d_map, map_max_row, map_max_col, map_resolution,
        d_circles, n_circles, d_rects, n_rects, Nr, dim_u, dim_x, Tr,
        (double)dt, gamma_u, model_type);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> hcr(Nr), hri((size_t)Nr * dim_u * Tr);
    CUDA_CHECK(cudaMemcpy(hcr.data(), d_costs_r, Nr * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hri.data(), d_Uri,
                          (size_t)Nr * dim_u * Tr * sizeof(double),
                          cudaMemcpyDeviceToHost));
    Eigen::VectorXd costs_r = Eigen::Map<Eigen::VectorXd>(hcr.data(), Nr);
    double mc = costs_r.minCoeff();
    Eigen::VectorXd wts = (-gamma_u * (costs_r.array() - mc)).exp();
    const double weight_sum = wts.sum();
    if (std::isfinite(weight_sum) && weight_sum > 0.0) {
        wts /= weight_sum;
    } else {
        wts = Eigen::VectorXd::Constant(Nr, 1.0 / static_cast<double>(Nr));
    }

    Eigen::MatrixXd Ures = Eigen::MatrixXd::Zero(dim_u, Tr);
    for (int i = 0; i < Nr; ++i) {
        for (int d = 0; d < dim_u; ++d) {
            for (int t = 0; t < Tr; ++t) {
                Ures(d, t) += wts(i) * hri[i * (dim_u * Tr) + d * Tr + t];
            }
        }
    }
    h(Ures);

    Eigen::MatrixXd Xi(dim_x, Tr + 1);
    Xi.col(0) = x_init;
    double cost = 0.0;
    for (int t = 0; t < Tr; ++t) {
        Xi.col(t + 1) = Xi.col(t) + (double)dt * f(Xi.col(t), Ures.col(t));
        cost += p(Xi.col(t), x_target);
    }
    cost += p(Xi.col(Tr), x_target);
    for (int t = 0; t < Tr + 1; ++t) {
        if (collision_checker && collision_checker->getCollisionGrid(Xi.col(t))) {
            cost = 1e8;
            break;
        }
    }

    bool ref_feasible = true;
    double ref_cost = 0.0;
    for (int t = 0; t < Xref.cols(); ++t) {
        if (collision_checker && collision_checker->getCollisionGrid(Xref.col(t))) {
            ref_feasible = false;
            break;
        }
        ref_cost += p(Xref.col(t), x_target);
    }

    Ur.clear();
    Cr.clear();
    Xr.clear();
    if (cost >= 1e7 && ref_feasible) {
        Uo = Uref;
        Xo = Xref;
        u0 = Uo.col(0);
        Ur.push_back(Uref);
        Cr.push_back(ref_cost);
        Xr.push_back(Xref);
    } else {
        Uo = Ures;
        Xo = Xi;
        u0 = Uo.col(0);
        Ur.push_back(Ures);
        Cr.push_back(cost);
        Xr.push_back(Xi);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    elapsed_guide += std::chrono::duration<double>(t1 - t0).count();
}

void BiMPPI_GPU::partitioningControl() {
    U_f0=Uo.leftCols(Tf);
    U_b0=Eigen::MatrixXd::Zero(dim_u,Tb);
}

void BiMPPI_GPU::solve() {
    elapsed_rollout=elapsed_clustering=0.0;
    vis_rollout_samples.clear();
    start=std::chrono::high_resolution_clock::now();
    backwardRollout(); forwardRollout();
    auto t2=std::chrono::high_resolution_clock::now();
    selectConnection(); concatenate();
    auto t3=std::chrono::high_resolution_clock::now();
    elapsed_connection=std::chrono::duration<double>(t3-t2).count();
    guideMPPI();
    auto t4=std::chrono::high_resolution_clock::now();
    elapsed_guide=std::chrono::duration<double>(t4-t3).count();
    elapsed_1=t4-start;
    partitioningControl();
    elapsed=elapsed_rollout+elapsed_clustering+elapsed_connection+elapsed_guide;

    // ── Visualization data export ──
    if (vis_logger && vis_logger->enabled) {
      if (!vis_rollout_samples.empty()) {
        vis_logger->saveTrajectories("rollouts", vis_rollout_samples);
      }
      // Forward cluster trajectories (Xf: clusters_f.size() * dim_x rows)
      std::vector<Eigen::MatrixXd> fwd_trajs;
      for (int ci = 0; ci < (int)clusters_f.size(); ++ci)
        fwd_trajs.push_back(Xf.block(ci * dim_x, 0, dim_x, Tf + 1));
      vis_logger->saveTrajectories("forward_clusters", fwd_trajs);
      // Backward cluster trajectories
      std::vector<Eigen::MatrixXd> bwd_trajs;
      for (int ci = 0; ci < (int)clusters_b.size(); ++ci)
        bwd_trajs.push_back(Xb.block(ci * dim_x, 0, dim_x, Tb + 1));
      vis_logger->saveTrajectories("backward_clusters", bwd_trajs);
      // All guide candidate trajectories
      if (!Xr.empty())
        vis_logger->saveTrajectories("guide_candidates", Xr);
      // Optimal trajectory
      vis_logger->saveTrajectory("optimal", Xo);
      vis_logger->savePosition(x_init);
    }

    visual_traj.push_back(x_init);
}

void BiMPPI_GPU::move() {
    x_init=x_init+(double)dt*f(x_init, u0);
    U_f0.leftCols(U_f0.cols()-1)=U_f0.rightCols(U_f0.cols()-1);
}
