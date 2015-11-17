#pragma once

struct sliceHelper {
    sliceHelper(const slice& index, ptrdiff_t n) : arraySize(n) {
        extract<ptrdiff_t> stepIn(index.step());
        if (stepIn.check())
            step = stepIn();
        else
            step = 1;

        if (step == 0) throw std::invalid_argument("step = 0 is not valid");

        extract<ptrdiff_t> startIn(index.start());
        if (startIn.check()) {
            if (step > 0) {
                start = startIn();
                if (start < 0) start += n;
            } else {
                start = startIn() + 1;
                if (start <= 0) start += n;
            }
        } else if (step > 0)
            start = 0;
        else
            start = n;

        extract<ptrdiff_t> stopIn(index.stop());
        if (stopIn.check()) {
            if (step > 0) {
                stop = stopIn();
                if (stop < 0) stop += n;
            } else {
                stop = stopIn() + 1;
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
                std::max<ptrdiff_t>(0, 1 + (start - stop - 1) / std::abs(step));

        if (start < 0 || stop > arraySize)
            throw std::out_of_range("Slice index out of range");
    }

    friend std::ostream& operator<<(std::ostream& ss, const sliceHelper& sh) {
        return ss << "Slice, start: " << sh.start << ", stop: " << sh.stop
                  << ", step: " << sh.step << ", numel: " << sh.numel;
    }
    ptrdiff_t start, stop, step, numel, arraySize;
};
