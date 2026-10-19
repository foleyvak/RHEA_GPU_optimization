#include "DistributedArray.hpp"
#include "ParallelTopology.hpp"

DistributedArray::DistributedArray(ParallelTopology* topo, char const* auxName) {
    size = topo->getSize();
    //vector = new double[size];
    vector = (double*) calloc ( size, sizeof ( double ) );
    ini_x = topo->iter_common[_INNER_][_INIX_] ;
    ini_y = topo->iter_common[_INNER_][_INIY_] ;
    ini_z = topo->iter_common[_INNER_][_INIZ_] ;
    fin_x = topo->iter_common[_INNER_][_ENDX_] ;
    fin_y = topo->iter_common[_INNER_][_ENDY_] ;
    fin_z = topo->iter_common[_INNER_][_ENDZ_] ;

    mydomain = topo;  

    sprintf(fieldName,"%s",auxName);
    
}

void DistributedArray::setTopology(ParallelTopology* topo, char const * auxName) {

    size = topo->getSize();
    //vector = new double[size];
    vector = (double*) calloc ( size, sizeof ( double ) );
    ini_x = topo->iter_common[_INNER_][_INIX_] ;
    ini_y = topo->iter_common[_INNER_][_INIY_] ;
    ini_z = topo->iter_common[_INNER_][_INIZ_] ;
    fin_x = topo->iter_common[_INNER_][_ENDX_] ;
    fin_y = topo->iter_common[_INNER_][_ENDY_] ;
    fin_z = topo->iter_common[_INNER_][_ENDZ_] ;

    mydomain = topo;   

    sprintf(fieldName,"%s",auxName);

    // Allocate and copy to GPU
    #pragma acc enter data copyin(vector[0:size])	// ... added for OpenACC

}

DistributedArray::~DistributedArray() {		// ... added for OpenACC
    #pragma acc exit data delete(vector[0:size])
    free(vector); // Free host memory
}

void DistributedArray::update() {
    #pragma acc data present(vector[0:size]) 
    {	
        //mydomain->update(vector);
#if _GPU_AWARE_MPI_DEACTIVATED_
        mydomain->update(vector);
#else
	mydomain->update_gpu(vector);
#endif
        fillEdgeCornerBoundaries();
    }
}

void DistributedArray::update_simple() {
    mydomain->update_simple(vector);
    fillEdgeCornerBoundaries();
}


double& DistributedArray::operator[](int idx) {
    return vector[idx];
}

void DistributedArray::operator=(double val) {
    for(int l = 0; l < size; l++)
        vector[l] = val;
}

void DistributedArray::fillEdgeCornerBoundaries() {
    #pragma acc data present(vector[0:size])
    {
        mydomain->fillEdgesCorners(vector);
    }
}
