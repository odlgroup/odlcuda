#pragma once

//Abstract base class
template<typename T>
class DeviceVector;

template<class T> using DeviceVectorPtr = std::shared_ptr<DeviceVector<T>>;