// Layer 2 functions: they use upstream internal C++ classes. Each one is in the Layer 2 register
// (brancusi-desktop _docs/sd-cpp-interface.md section 5.3). Layer 1 calls them through this header,
// which has no upstream internal types.
#ifndef BSD_L2_H
#define BSD_L2_H

#include <string>

#include "bsd.h"

namespace bsd_l2 {

// Read the headers of the given parts with upstream ModelLoader. `wtype` is an sd_type_t value
// for the memory estimate, or -1 to keep the file types. On failure, sets `error`.
bsd_status_t inspect(const bsd_components_t* parts, int wtype, bsd_inspect_t* out, std::string& error);

}  // namespace bsd_l2

#endif  // BSD_L2_H
