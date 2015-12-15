#pragma once

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <iostream>

struct sliceHelper {
    sliceHelper(const py::slice& index, ptrdiff_t n) : arraySize(n) {
		index.compute(n, &start, &stop, &step, &numel);
		std::cout << start << " " << stop << " " << step << " " << numel << std::endl;

		//TODO make consistient with python
		if (step < 0){
			start += 1;
			stop += 1;
		}

		//if (step == 0)
		//	step = 1;

        /*if (start != 0) {
            if (step > 0) {
                if (start < 0) start += n;
            } else {
                start += 1;
                if (start <= 0) start += n;
            }
        } else if (step > 0)
            start = 0;
        else
            start = n;

        if (stop != 0) {
            if (step > 0) {
                if (stop < 0) stop += n;
            } else {
                stop += 1;
                if (stop <= 0) stop += n;
            }
        } else if (step > 0)
            stop = n;
        else
            stop = 0;

        if (start == stop)
            numel = 0;
        else if (step > 0)
            numel = std::max<ptrdiff_t>(0, 1 + (stop - start - 1) / step);
        else
            numel =
                std::max<ptrdiff_t>(0, 1 + (start - stop - 1) / std::abs(step));*/

        if (start < 0 || stop > arraySize)
            throw std::out_of_range("Slice index out of range");
    }

    friend std::ostream& operator<<(std::ostream& ss, const sliceHelper& sh) {
        return ss << "Slice, start: " << sh.start << ", stop: " << sh.stop
                  << ", step: " << sh.step << ", numel: " << sh.numel;
    }
    ptrdiff_t start, stop, step, numel, arraySize;
};
