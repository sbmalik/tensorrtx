#include "src/utils.h"

int main(int argc, char **argv) {
    std::map <std::string, Weights> weightsVals = loadWeights("../xtestnet.wts");
    printWeightKeys(&weightsVals);
}