#include "misc.h"


PYBIND11_MODULE(xerus, m) {
    // m.doc() = "...";

    // xerus version
    m.attr("VERSION_MAJOR") = VERSION_MAJOR;
    m.attr("VERSION_MINOR") = VERSION_MINOR;
    m.attr("VERSION_REVISION") = VERSION_REVISION;
    m.attr("VERSION_COMMIT") = VERSION_COMMIT;

    expose_indexedTensors(m);
//    expose_factorizations(m);

    expose_tensor(m);
    expose_tensorNetwork(m);
    expose_ttnetwork(m);
    expose_htnetwork(m);

    /* expose_leastSquaresAlgorithms(m); */
    /* expose_recoveryAlgorithms(m); */

    expose_misc(m);
}
