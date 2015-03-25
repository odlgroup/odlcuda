#pragma once

#include <boost/python.hpp>
#include <RLcpp/function_traits.h>
#include <type_traits>
#include <utility>

namespace boost {
namespace python {
namespace detail {
template <class T, class... Args>
inline boost::mpl::vector<T, Args...>
get_signature(std::function<T(Args...)>, void* = 0) {
    return boost::mpl::vector<T, Args...>();
}
}
}
}

template <typename T>
struct cpy {
    static typename RetType<decltype(copyInput<T>)>::type
    apply(const typename ArgType<decltype(copyInput<T>)>::type& x) {
        return copyInput<T>(x);
    }
};

template <typename T>
struct map {
    static typename RetType<decltype(mapInput<T>)>::type
	apply(const typename ArgType<decltype(mapInput<T>)>::type& x) {
		return mapInput<T>(x);
    }
};

template <typename T>
struct id {
    static T& apply(T& v) { return v; }
    static const T& apply(const T& v) { return v; }

    static auto apply(T&& v) -> decltype(std::forward<T>(v)) {
        return std::forward<T>(v);
    }
};

template <typename T>
T& identity(T& v) { return v; }

template <typename T>
const T& itentity(const T& v) { return v; }

template <typename T>
auto itentity(T&& v) -> decltype(std::forward<T>(v)) { return std::forward<T>(v); }

template <typename F, typename O, typename... T>
std::function<typename RetType<O>::type(typename ArgType<T>::type&&...)>
wrap(F f, O o, T... ts) {
    return [=](typename ArgType<T>::type&&... args) { 
		return o(f(ts(std::forward<typename ArgType<T>::type>(args))...));
    };
}

template <typename Base, typename... T>
struct ClassWrapper : public Base {
    template <typename... Args>
    ClassWrapper(Args... args) : Base(T::apply(args)...) {}
};

template <typename C, typename F, typename O, typename... T>
std::function<typename RetType<O>::type(C& c, typename ArgType<T>::type&&...)>
wrapMember(F f, O o, T... ts) {
    return [=](C& c, typename ArgType<T>::type&&... args) {
		return o((c.*f)(ts(std::forward<typename ArgType<T>::type>(args))...));
    };
}

template <typename C, typename F, typename... T>
std::function<void(C& c, typename ArgType<T>::type&&...)>
wrapVoidMember(F f, T... ts) {
    return [=](C& c, typename ArgType<T>::type&&... args) {
		(c.*f)(ts(std::forward<typename ArgType<T>::type>(args))...);
    };
}