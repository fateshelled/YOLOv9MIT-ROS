#pragma once
#define CV_BRIDGE_VERSION_GTE(major, minor, patch) \
    ((major < CV_BRIDGE_VERSION_MAJOR)   ? true    \
     : (major > CV_BRIDGE_VERSION_MAJOR) ? false   \
     : (minor < CV_BRIDGE_VERSION_MINOR) ? true    \
     : (minor > CV_BRIDGE_VERSION_MINOR) ? false   \
     : (patch < CV_BRIDGE_VERSION_PATCH) ? true    \
     : (patch > CV_BRIDGE_VERSION_PATCH) ? false   \
                                         : true)

#if CV_BRIDGE_VERSION_GTE(3, 4, 0)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
