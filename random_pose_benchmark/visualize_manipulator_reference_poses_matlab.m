function visualize_manipulator_reference_poses_matlab()
%VISUALIZE_MANIPULATOR_REFERENCE_POSES_MATLAB
% Visualize the 16 joint-space reference poses used by the manipulator
% random-pose benchmark on the MathWorks predefined Universal Robots UR5
% model.
%
% The benchmark poses are 6-DOF joint postures [rad]. For the MATLAB UR5 URDF
% convention, joint 1 is shifted by pi to align the base direction with the
% benchmark DH convention:
%   ur5 = loadrobot('universalUR5', 'DataFormat', 'row');
%   show(ur5, benchmarkPoseToUr5Configuration(q));
%
% Usage from MATLAB:
%   cd /home/cora-tuf/MPPI/BiC-MPPI-legacy/random_pose_benchmark
%   visualize_manipulator_reference_poses_matlab

close all;

[poses, names, ids] = referencePoses();
dh = manipulatorDhParameters();
figurePosition = [80, 80, 1100, 1100];
workspace = [-0.85, 0.35, -1.05, 0.75, -0.05, 0.95];
subplotWidthScale = 0.88;   % Smaller value gives more horizontal gap.
subplotHeightScale = 0.74;  % Smaller value gives more vertical gap.
titleFontSize = 8;
labelFontSize = 7;
tickFontSize = 7;

robot = [];
if hasLoadrobot()
    try
        robot = loadUniversalUR5Robot();
    catch ME
        warning(['Could not load MathWorks predefined Universal UR5 model: %s\n', ...
                 'Using the local DH skeleton plot only.'], ME.message);
    end
else
    warning(['Robotics System Toolbox loadrobot was not found. ', ...
             'Using the local DH skeleton plot only.']);
end

fig = figure('Name', 'Universal UR5 Reference Poses', ...
             'Color', 'w', ...
             'Position', figurePosition);

for i = 1:size(poses, 1)
    figure(fig);
    ax = subplot(4, 4, i);
    benchmarkQ = poses(i, :);

    shownWithToolbox = false;
    if ~isempty(robot)
        ur5Q = benchmarkPoseToUr5Configuration(benchmarkQ);
        shownWithToolbox = showRigidBodyTree(ax, robot, ur5Q);
    end

    hold(ax, 'on');
    if ~shownWithToolbox
        plotRigidBodyGeometry(ax, benchmarkQ, dh);
        plotDhCenterline(ax, benchmarkQ, dh);
    end
    hold(ax, 'off');

    title(ax, sprintf('%02d  %s', ids(i), names{i}), ...
          'Interpreter', 'none', 'FontSize', titleFontSize);
    xlabel(ax, 'x [m]', 'FontSize', labelFontSize);
    ylabel(ax, 'y [m]', 'FontSize', labelFontSize);
    zlabel(ax, 'z [m]', 'FontSize', labelFontSize);
    ax.FontSize = tickFontSize;
    xlim(ax, workspace(1:2));
    ylim(ax, workspace(3:4));
    zlim(ax, workspace(5:6));
    axis(ax, 'equal');
    grid(ax, 'on');
    view(ax, 135, 25);
    applyLighting(ax);
    adjustSubplotPosition(ax, subplotWidthScale, subplotHeightScale);
end

outDir = fullfile(fileparts(mfilename('fullpath')), 'results');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
outPng = fullfile(outDir, 'manipulator_reference_poses_matlab_ur5.png');
saveFigurePng(fig, outPng);
fprintf('Saved figure: %s\n', outPng);

end

function tf = hasLoadrobot()
tf = exist('loadrobot', 'file') == 2;
end

function robot = loadUniversalUR5Robot()
robot = loadrobot('universalUR5', 'DataFormat', 'row');
numJoints = numel(homeConfiguration(robot));
if numJoints ~= 6
    error('Expected universalUR5 to have 6 non-fixed joints, but got %d.', numJoints);
end
end

function ur5Q = benchmarkPoseToUr5Configuration(benchmarkQ)
ur5Q = benchmarkQ;
ur5Q(1) = ur5Q(1) + pi;
ur5Q = wrapToPiLocal(ur5Q);
end

function q = wrapToPiLocal(q)
q = mod(q + pi, 2.0 * pi) - pi;
end

function adjustSubplotPosition(ax, widthScale, heightScale)
pos = ax.Position;
pos(1) = pos(1) + 0.5 * pos(3) * (1.0 - widthScale);
pos(2) = pos(2) + 0.5 * pos(4) * (1.0 - heightScale);
pos(3) = pos(3) * widthScale;
pos(4) = pos(4) * heightScale;
ax.Position = pos;
end

function dh = manipulatorDhParameters()
dh.a = [0.0, -0.427, -0.357, 0.0, 0.0, 0.0];
dh.d = [0.15, 0.0, 0.0, 0.11, 0.09, 0.09];
dh.alpha = [pi / 2.0, 0.0, 0.0, pi / 2.0, -pi / 2.0, 0.0];
end

function robot = buildRigidBodyTree(dh)
robot = rigidBodyTree('DataFormat', 'row', 'MaxNumBodies', 6);
parent = robot.BaseName;
linkRadius = [0.050, 0.045, 0.040, 0.032, 0.028, 0.024];
jointRadius = [0.060, 0.052, 0.048, 0.040, 0.034, 0.030];

for i = 1:6
    body = rigidBody(sprintf('link%d', i));
    joint = rigidBodyJoint(sprintf('joint%d', i), 'revolute');
    joint.JointAxis = [0, 0, 1];

    % Robotics System Toolbox uses standard DH order [a alpha d theta].
    setFixedTransform(joint, [dh.a(i), dh.alpha(i), dh.d(i), 0.0], 'dh');
    body.Joint = joint;
    addOfficialBodyVisuals(body, i, dh, linkRadius(i), jointRadius(i));

    addBody(robot, body, parent);
    parent = body.Name;
end
end

function addOfficialBodyVisuals(body, i, dh, linkRadius, jointRadius)
linkColor = [0.78, 0.83, 0.90];
jointColor = [0.12, 0.28, 0.52];
eeColor = [0.86, 0.18, 0.12];

T0 = standardDhTransform(0.0, dh.a(i), dh.d(i), dh.alpha(i));
parentInBody = T0 \ [0.0; 0.0; 0.0; 1.0];
p0 = [0.0, 0.0, 0.0];
p1 = parentInBody(1:3).';
linkLength = norm(p1 - p0);

if linkLength > 1.0e-8
    linkTform = visualTformForSegment(p0, p1);
    addVisual(body, 'Capsule', [linkRadius, linkLength], linkTform, ...
              'FaceColor', linkColor, 'FaceAlpha', 0.96);
end

jointColorThis = jointColor;
if i == 6
    jointColorThis = eeColor;
end
addVisual(body, 'Sphere', jointRadius, eye(4), ...
          'FaceColor', jointColorThis, 'FaceAlpha', 0.98);

if i == 6
    addVisual(body, 'Capsule', [0.018, 0.095], ...
              makeTranslationTform([0.0475, 0.0, 0.0]) * makeYRotationTform(pi / 2.0), ...
              'FaceColor', eeColor, 'FaceAlpha', 0.98);
end
end

function T = visualTformForSegment(p0, p1)
v = p1 - p0;
len = norm(v);
ez = v / len;
tmp = [0.0, 0.0, 1.0];
if abs(dot(ez, tmp)) > 0.95
    tmp = [0.0, 1.0, 0.0];
end
ex = cross(tmp, ez);
ex = ex / norm(ex);
ey = cross(ez, ex);
mid = 0.5 * (p0 + p1);

T = eye(4);
T(1:3, 1) = ex(:);
T(1:3, 2) = ey(:);
T(1:3, 3) = ez(:);
T(1:3, 4) = mid(:);
end

function T = makeTranslationTform(p)
T = eye(4);
T(1:3, 4) = p(:);
end

function T = makeYRotationTform(theta)
c = cos(theta);
s = sin(theta);
T = [ c, 0.0,  s, 0.0; ...
      0.0, 1.0, 0.0, 0.0; ...
     -s, 0.0,  c, 0.0; ...
      0.0, 0.0, 0.0, 1.0];
end

function shown = showRigidBodyTree(ax, robot, q)
shown = false;
try
    show(robot, q, ...
         'Parent', ax, ...
         'PreservePlot', false, ...
         'Frames', 'off', ...
         'Visuals', 'on', ...
         'Collisions', 'off');
    shown = true;
catch
    try
        show(robot, q, 'Parent', ax, 'PreservePlot', false, 'Visuals', 'on');
        shown = true;
    catch
        shown = false;
    end
end
end

function plotDhSkeleton(ax, q, dh)
pts = fkJointPositions(q, dh);

plot3(ax, pts(:, 1), pts(:, 2), pts(:, 3), ...
      '-', 'LineWidth', 2.2, 'Color', [0.05, 0.22, 0.55]);
scatter3(ax, pts(:, 1), pts(:, 2), pts(:, 3), ...
         24, [0.08, 0.33, 0.72], 'filled');
scatter3(ax, pts(1, 1), pts(1, 2), pts(1, 3), ...
         42, [0.05, 0.05, 0.05], 'filled');
scatter3(ax, pts(end, 1), pts(end, 2), pts(end, 3), ...
         54, [0.84, 0.15, 0.12], 'filled');
end

function plotRigidBodyGeometry(ax, q, dh)
pts = fkJointPositions(q, dh);

linkRadius = [0.050, 0.045, 0.040, 0.032, 0.028, 0.024];
jointRadius = [0.060, 0.052, 0.048, 0.040, 0.034, 0.030, 0.030];
linkColor = [0.78, 0.83, 0.90];
linkEdgeColor = [0.40, 0.47, 0.58];
jointColor = [0.12, 0.28, 0.52];
baseColor = [0.18, 0.18, 0.18];
eeColor = [0.86, 0.18, 0.12];

drawCylinderBetween(ax, [0.0, 0.0, -0.035], [0.0, 0.0, 0.065], ...
                    0.080, baseColor, baseColor, 20);

for i = 1:6
    drawCylinderBetween(ax, pts(i, :), pts(i + 1, :), ...
                        linkRadius(i), linkColor, linkEdgeColor, 18);
end

for i = 1:7
    if i == 1
        color = baseColor;
    elseif i == 7
        color = eeColor;
    else
        color = jointColor;
    end
    drawSphere(ax, pts(i, :), jointRadius(i), color, 16);
end

drawEndEffectorTool(ax, q, dh, eeColor);
end

function plotDhCenterline(ax, q, dh)
pts = fkJointPositions(q, dh);
plot3(ax, pts(:, 1), pts(:, 2), pts(:, 3), ...
      '-', 'LineWidth', 1.1, 'Color', [0.04, 0.12, 0.28]);
end

function drawEndEffectorTool(ax, q, dh, color)
T = eye(4);
for i = 1:6
    T = T * standardDhTransform(q(i), dh.a(i), dh.d(i), dh.alpha(i));
end

p0 = T(1:3, 4).';
toolAxis = T(1:3, 1).';
if norm(toolAxis) < 1.0e-12
    toolAxis = [1.0, 0.0, 0.0];
end
toolAxis = toolAxis / norm(toolAxis);
p1 = p0 + 0.095 * toolAxis;
drawCylinderBetween(ax, p0, p1, 0.018, color, color, 14);
drawSphere(ax, p1, 0.026, color, 14);
end

function drawCylinderBetween(ax, p0, p1, radius, faceColor, edgeColor, n)
v = p1 - p0;
len = norm(v);
if len < 1.0e-10
    drawSphere(ax, p0, radius, faceColor, n);
    return;
end

ez = v / len;
tmp = [0.0, 0.0, 1.0];
if abs(dot(ez, tmp)) > 0.95
    tmp = [0.0, 1.0, 0.0];
end
ex = cross(tmp, ez);
ex = ex / norm(ex);
ey = cross(ez, ex);

[X, Y, Z] = cylinder(radius, n);
Z = Z * len;

Xw = p0(1) + ex(1) * X + ey(1) * Y + ez(1) * Z;
Yw = p0(2) + ex(2) * X + ey(2) * Y + ez(2) * Z;
Zw = p0(3) + ex(3) * X + ey(3) * Y + ez(3) * Z;

surf(ax, Xw, Yw, Zw, ...
     'FaceColor', faceColor, ...
     'EdgeColor', edgeColor, ...
     'EdgeAlpha', 0.18, ...
     'FaceAlpha', 0.96);
end

function drawSphere(ax, center, radius, color, n)
[X, Y, Z] = sphere(n);
surf(ax, center(1) + radius * X, ...
         center(2) + radius * Y, ...
         center(3) + radius * Z, ...
     'FaceColor', color, ...
     'EdgeColor', 'none', ...
     'FaceAlpha', 0.98);
end

function applyLighting(ax)
try
    lighting(ax, 'gouraud');
    material(ax, 'dull');
    camlight(ax, 'headlight');
catch
end
end

function pts = fkJointPositions(q, dh)
T = eye(4);
pts = zeros(7, 3);
pts(1, :) = [0.0, 0.0, 0.0];

for i = 1:6
    T = T * standardDhTransform(q(i), dh.a(i), dh.d(i), dh.alpha(i));
    pts(i + 1, :) = T(1:3, 4).';
end
end

function A = standardDhTransform(theta, a, d, alpha)
ct = cos(theta);
st = sin(theta);
ca = cos(alpha);
sa = sin(alpha);

A = [ct, -st * ca,  st * sa, a * ct; ...
     st,  ct * ca, -ct * sa, a * st; ...
     0.0,       sa,       ca,      d; ...
     0.0,      0.0,      0.0,    1.0];
end

function saveFigurePng(fig, outPng)
try
    exportgraphics(fig, outPng, 'Resolution', 200);
catch
    print(fig, outPng, '-dpng', '-r200');
end
end

function [poses, names, ids] = referencePoses()
ids = (0:15).';
names = { ...
    'home_mid'; ...
    'detour_high'; ...
    'goal_right'; ...
    'pre_pick'; ...
    'pick_low'; ...
    'transfer_high'; ...
    'pre_place'; ...
    'place_low'; ...
    'left_high'; ...
    'left_low'; ...
    'right_high'; ...
    'right_low'; ...
    'over_left'; ...
    'over_right'; ...
    'front_left_mid'; ...
    'front_right_mid'};

poses = [ ...
     0.05, -1.15, 1.25,  0.00, 0.85,  0.00; ...
     1.20, -1.40, 0.50, -0.20, 0.70,  0.10; ...
     1.45, -0.80, 1.05, -0.20, 0.65,  0.25; ...
    -0.65, -1.18, 1.22,  0.00, 0.78,  0.00; ...
    -0.65, -0.82, 1.52,  0.00, 0.38,  0.00; ...
     0.10, -1.45, 1.25,  0.00, 1.05,  0.00; ...
     0.90, -1.18, 1.18,  0.00, 0.80,  0.00; ...
     0.90, -0.78, 1.52,  0.00, 0.36,  0.00; ...
    -0.90, -1.25, 1.10,  0.15, 0.95,  0.05; ...
    -0.90, -0.90, 1.45,  0.10, 0.45,  0.05; ...
     1.10, -1.25, 1.10, -0.15, 0.95, -0.05; ...
     1.10, -0.90, 1.45, -0.10, 0.45, -0.05; ...
    -0.35, -1.55, 1.35,  0.10, 1.10,  0.00; ...
     0.35, -1.55, 1.35, -0.10, 1.10,  0.00; ...
    -0.45, -0.95, 1.30,  0.10, 0.65,  0.10; ...
     0.45, -0.95, 1.30, -0.10, 0.65, -0.10];
end
