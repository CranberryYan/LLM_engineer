add_library(layerweights STATIC layer_weights.cpp)
target_link_libraries(layerweights PUBLIC -lcudart weightutils)

add_library(llamaweights STATIC llama_weights.cc)
target_link_libraries(llamaweights PUBLIC -lcudart weightutils)