// ------------------------------------------------------------------
// This copy of code is originally written by Ross Girshick
// I adjust it to 3D case and new version of caffe, also make it no weights.
// Jan, 2017
// ------------------------------------------------------------------

#include "caffe/layers/smoothL1_layer.hpp"

namespace caffe {

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //has_weights_ = (bottom.size() == 3);//by dongnie
  has_weights_=0;
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));//1 means channels bydong
  CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(2));
  CHECK_EQ(bottom[0]->shape(3), bottom[1]->shape(3));
  CHECK_EQ(bottom[0]->shape(4), bottom[1]->shape(4));
  if (has_weights_) {
    CHECK_EQ(bottom[0]->shape(1), bottom[2]->shape(1));
    CHECK_EQ(bottom[0]->shape(2), bottom[2]->shape(2));
    CHECK_EQ(bottom[0]->shape(3), bottom[2]->shape(3));
    CHECK_EQ(bottom[0]->shape(4), bottom[2]->shape(4));
  }

  //diff_.Reshape(bottom[0]->shape(0), bottom[0]->shape(1),
  //    bottom[0]->shape(2), bottom[0]->shape(3), bottom[0]->shape(4));
  //errors_.Reshape(bottom[0]->shape(0), bottom[0]->shape(1),
  //    bottom[0]->shape(2), bottom[0]->shape(3),bottom[0]->shape(4));
  diff_.ReshapeLike(*bottom[0]);//by dong 
  errors_.ReshapeLike(*bottom[0]); //by dong
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(SmoothL1LossLayer);
#endif

INSTANTIATE_CLASS(SmoothL1LossLayer);
REGISTER_LAYER_CLASS(SmoothL1Loss);

}  // namespace caffe
