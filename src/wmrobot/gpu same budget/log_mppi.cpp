#include "same_budget_common.h"

int main(int argc, char **argv) {
  return runSameBudgetExperiment(
      argc, argv, sameBudgetSelectedMethods(SameBudgetMethod::LogMPPI));
}
