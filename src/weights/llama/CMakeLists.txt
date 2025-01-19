add_library(layerweights STATIC layer_weights.cpp)
target_link_libraries(layerweights PUBLIC -lcudart weightutils)