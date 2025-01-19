# pragma once
# include <vector>
# include <string>
# include <numeric>
# include <sstream>
# include <iostream>
# include <algorithm>
# include <unordered_map>
# include <cuda_fp16.h>
# include "src/utils/macro.h"
# include "src/utils/string.h"

enum Device {
	CPU_PINNED,
	CPU,
	GPU
};

enum DataType {
	FP32,
	FP16,
	INT8,
	INT32,
	BOOL,
	BYTES,
	UNSUPPORTED
};

template<typename T>
DataType getTensorType() {
	if (std::is_same<T, float>::value || std::is_same<T, const float>::value) {
		return FP32;
	}
	else if (std::is_same<T, half>::value || std::is_same<T, const half>::value) {
		return FP16;
	}
	else if (std::is_same<T, int>::value || std::is_same<T, const int>::value) {
		return INT32;
	}
	else if (std::is_same<T, int8_t>::value || std::is_same<T, const int8_t>::value) {
		return INT8;
	}
	else if (std::is_same<T, bool>::value || std::is_same<T, const bool>::value) {
		return BOOL;
	}
	else if (std::is_same<T, char>::value || std::is_same<T, const char>::value) {
		return BYTES;
	}
	else {
		return UNSUPPORTED;
	}
}

// 提前声明, Tensor->as要用
template<typename T>
class TensorWrapper;

// data会被放在TensorWrapper中, 想对TensorWrapper进行模板化
struct Tensor {
	// 默认构造函数
	Tensor() = default;

	// 带参构造(初始化)
	Tensor(
		const Device location_, 
		const DataType dtype_,
		const std::vector<int> shape_) :
		location(location_),
		dtype(dtype_),
		shape(shape_) {}

	// 虚函数, 可以重写也可以不重写(纯虚函数必须重写)
	virtual int size() const {
		if (shape.size() == 0) {
			return 0;
		}

		// 逐元素乘
		return std::accumulate(shape.begin(), shape.end(),
			(int)1, std::multiplies<int>());
	}

	// 下行转换, 父 -> 子
	// 将当前对象转换为 TensorWrapper<T> 类型的指针
	// static_cast: dynamic是最安全的，但是使用static,
	//	因为我们可以确保这个tensor就是TensorWrapper
	template<typename T>
	TensorWrapper<T> *as() {
		return static_cast<TensorWrapper<T>*>(this);
	}

	// 返回tensor所在位置(host or device)
	std::string DeviceString() const {
		static const std::unordered_map<Device, std::string> devicetring {
			{CPU, "CPU"}, {CPU_PINNED, "CPU_PINNED"}, {GPU, "GPU"}};
			return devicetring.at(location);
	}

	// toString
	virtual std::string toString() const {
		std::string device_str = DeviceString();
		static const std::unordered_map<DataType, std::string> type_to_string {
			{INT8, "INT8"}, {INT32,"INT32"}, {FP16, "FP16"}, {FP32, "FP32"}};
			return fmtstr("Tensor[where=%s, type=%s, shape=%s]",
				device_str.c_str(),
				type_to_string.at(dtype).c_str(),
				vec2str(shape).c_str());
	}

	Device location;
	DataType dtype;
	std::vector<int> shape;
};

template<typename T>
class TensorWrapper: public Tensor {
public:
	// 调用基类来初始化(无 data)
	TensorWrapper(Device location, DataType dtype, std::vector<int> shape):
		Tensor(location, dtype, shape) {}

	// 调用基类来初始化(有 data)
	TensorWrapper(Device location, DataType dtype,
		std::vector<int> shape, T *data):
		Tensor(location, dtype, shape), data(data) {
			DataType in_dtype = getTensorType<T>();
			LLM_CHECK_WITH_INFO(in_dtype == dtype, 
				"the type should be same as dtype in params");
	}

	virtual int size() const {
		if (data == nullptr || shape.size() == 0) {
			return 0;
		}
		return std::accumulate(shape.begin(), shape.end(),
			(int)1, std::multiplies<int>());
	}

	inline T getVal(int id) const {
		LLM_CHECK(location == CPU);
		return data[id];
	}

	inline T getVal() const {
		LLM_CHECK(location == CPU);
		return getVal(0);
	}

	inline T *getPtr() const {
		return (T*) data;
	}

	inline std::string toString() const {
		std::string device_str = DeviceString();

	static const std::unordered_map<DataType, std::string> type_to_string {
		{INT8, "INT8"},
		{FP16, "FP16"},
		{FP32, "FP32"},};

	return fmtstr("Tensor[where=%s, type=%s, shape=%s, data=%p]",
								device_str.c_str(),
								type_to_string.at(dtype).c_str(),
								vec2str(shape).c_str(),
								data);
	}

public:
	T *data;
};

struct TensorMap {
	TensorMap() = default;

	// initializer_list: 自行指定std::pair<std::string, Tensor* >
	TensorMap(std::initializer_list<std::pair<std::string, Tensor* >> tensor_map) {
		for (const auto& pair : tensor_map) {
			if (isValid(pair.second)) {
				insert(pair.first, pair.second);
			} else {
				LLM_CHECK_WITH_INFO(isValid(pair.second),
					fmtstr("%s is not a valid tensor", pair.first.c_str()));
			}
		}
	}

	// unordered_map: 自带pair
	TensorMap(const std::unordered_map<std::string, Tensor* >& tensor_map) {
		for (const auto& pair : tensor_map) {
			if (isValid(pair.second)) {
				insert(pair.first, pair.second);
			} else {
				LLM_CHECK_WITH_INFO(isValid(pair.second), 
					fmtstr("%s is not a valid tensor", pair.first.c_str()));
			}
		}
	}

	~TensorMap() {
		tensor_map_.clear();
	}

	inline size_t size() const {
		return tensor_map_.size();
	}

	inline bool isExist(const std::string& key) const {
		return tensor_map_.find(key) != tensor_map_.end();
	}

	inline bool isValid(const Tensor* tensor) {
		return tensor->size() > 0;
	}

	// 增
	inline bool insert(const std::string& key, Tensor* value) {
		tensor_map_[key] = value;
		return true;
	}

	inline void insert(std::pair<std::string, Tensor* > p) {
		tensor_map_.insert(p);
	}

	// 删

	// 改

	// 查
	inline Tensor* at(const std::string& key) {
		// TODO: add a check to check key is existed
		LLM_CHECK_WITH_INFO(isExist(key), 
			fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
			key.c_str(),
			vec2str(keys()).c_str()));
		return tensor_map_.at(key);
	}

	inline Tensor* operator [] (const std::string &key) {
		LLM_CHECK_WITH_INFO(isExist(key), 
			fmtstr("Cannot find a tensor of name %s in the tensor map	(keys: %s)",
			key.c_str(),
			vec2str(keys()).c_str()));
		return tensor_map_.at(key);
	}

	//for debug
	std::vector<std::string> keys() const {
		std::vector<std::string> key_names;
		for (auto& kv : tensor_map_) {
				key_names.push_back(kv.first);
		}
		return key_names;
	}

	// 打印出tensormap中的所有key
	std::string toString() {
		std::stringstream ss;
		ss << "{";
		std::vector<std::string> key_names = keys();
		for (size_t i = 0; i < tensor_map_.size(); ++i) {
			ss << key_names[i] << ": " << at(key_names[i])->toString();
			if (i < tensor_map_.size() - 1) {
					ss << ", ";
			}
		}
		ss << "}";
		return ss.str();
	}

	// key: tensor的名字, val: tensor
	std::unordered_map<std::string, Tensor* > tensor_map_;
};
