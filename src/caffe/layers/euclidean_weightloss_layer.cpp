#include <vector>

#include "caffe/layers/euclidean_weightloss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/*template <typename Dtype>
void EuclideanLossWLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
		//weight=this->layer_param_.euclidean_lossw_param().weight();//must be in the hpp and in the protoxt
		
		
		
	}
*/


template <typename Dtype>
void EuclideanLossWLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  tmp_.ReshapeLike(*bottom[0]);//ew have to reshape it to tensor
}

template <typename Dtype>
void EuclideanLossWLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  const Dtype* weight_data = bottom[2]->cpu_data();//an extra blob which contains elementwise weight
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  //note diff_ is not right if we use weighted least square
  //const Dtype* tmp_cpu_data=diff_.cpu_data();//save conv7-gt
   
  //std::cout<<"tmp_cpu_data 0: "<<tmp_cpu_data[0]<<std::endl;
  caffe_mul(count,diff_.cpu_data(), weight_data, tmp_.mutable_cpu_data());//elementwise multipilication: w*(conv7-gt), then it is okay to compute the bp gradient
  //std::cout<<"tmp_cpu_data 1: "<<tmp_cpu_data[0]<<std::endl;
  //std::cout<<"diff_cpu_data 1: "<<diff_.cpu_data()[0]<<std::endl;
  
  //Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype weighted_dot = caffe_cpu_dot(count, tmp_.cpu_data(), diff_.cpu_data());
  //Dtype dist=exp(-1*dot);//distance weight, given weigh

  
  Dtype loss = weighted_dot / bottom[0]->shape(0) / Dtype(2);//shape(0) should be number of samples, we have weight at every location
  //loss=dist*loss;
   //this->blobs_.resize(1);//only one blob (the weight) in this way is trainable
   //this->blobs_[0].reset(new Blob<Dtype>(1));//only 1D
  top[0]->mutable_cpu_data()[0] = loss;
  //caffe_mul(count,diff_.cpu_data(),weight_data,diff_.mutable_cpu_data());//elementwise multipilication, then it is okay to compute the bp gradient
}

template <typename Dtype>
void EuclideanLossWLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight_data = bottom[2]->cpu_data();//an extra blob which contains elementwise weight
  int count = bottom[0]->count();
  std::cout<<"diff_cpu_data 1: "<<diff_.cpu_data()[0]<<std::endl;
  std::cout<<"weight_data 1: "<<weight_data[0]<<std::endl;
  caffe_mul(count, diff_.cpu_data(), weight_data, diff_.mutable_cpu_data());//elementwise multipilication: w*(conv7-gt), then it is okay to compute the bp gradient
  std::cout<<"diff_.cpu_data 1: "<<diff_.cpu_data()[0]<<std::endl; 
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->shape(0);
      
  	  std::cout<<"alpha :"<<alpha<<std::endl;
	  caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a, as it is already contain weight: weight*(conv7-gt)
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
  
  caffe_mul(count, diff_.cpu_data(), diff_.cpu_data(), bottom[2]->mutable_cpu_diff());//elementwise multipilication: w*(conv7-gt), then it is okay to compute the bp gradient
  std::cout<<"bottom[0]->cpu_diff 1: "<<bottom[0]->cpu_diff()[0]<<std::endl;
  std::cout<<"bottom[1]->cpu_diff 1: "<<bottom[1]->cpu_diff()[0]<<std::endl;
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanWLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossWLayer);
REGISTER_LAYER_CLASS(EuclideanLossW);

}  // namespace caffe
