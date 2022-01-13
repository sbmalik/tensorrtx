#include "src/xtestnet.h"
#include "NvInferPlugin.h"

int main(int argc, char **argv) {

    XTestNet xTestNet = XTestNet();
    if (argc == 2 && std::string(argv[1]) == "-s") {
        std::cout << "Serializling Engine" << std::endl;
        xTestNet.serializeEngine();
        return 0;
    } else if (argc == 2 && std::string(argv[1]) == "-d") {
        xTestNet.init();

        return 0;
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./xtestnet -s  // serialize model to plan file" << std::endl;
        std::cerr << "./xtestnet -d  // deserialize plan file and run inference" << std::endl;
        return -1;
    }
}
