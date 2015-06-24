#pragma once

template <class F>
struct ArgType;

template <class R, class T>
struct ArgType<R (*)(T)> : public ArgType<R(T)> {};

template <class R, class T>
struct ArgType<R(T)> {
    typedef T type;
};

template <class F>
struct RetType;

template <class R, class... T>
struct RetType<R (*)(T...)> : public RetType<R(T...)> {};

template <class F>
struct RetType;
template <class R, class... T>
struct RetType<R(T...)> {
    typedef R type;
};

template <class F>
struct ClassType;
template <class R, class C, class... T>
struct ClassType<R (C::*)(T...)> {
    typedef C type;
};
