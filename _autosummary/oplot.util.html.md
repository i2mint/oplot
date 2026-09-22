# oplot.util

Utils

### Functions

| [`cast_inputs`](#oplot.util.cast_inputs)(\*args_cast, \*\*kwargs_cast)         | Creates a decorator that casts the arguments and keyword arguments of a function according to specified casting functions.                                                      |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`fixed_step_chunker`](#oplot.util.fixed_step_chunker)(it, chk_size[, chk_step, ...]) | a function to get (an iterator of) segments (bt, tt) of chunks from an iterator (or list) of the for [it_1, it_2...], given a chk_size, chk_step, and a start_at and a stop_at. |
| [`float_to_timestamp`](#oplot.util.float_to_timestamp)(floats[, reference_time])      | Convert an array-like of floats (days since reference datetime) back to Timestamps.                                                                                             |
| [`timestamp_to_float`](#oplot.util.timestamp_to_float)(timestamps[, reference_time])  | Convert an array-like of Timestamps to absolute floats representing days since a fixed reference datetime.                                                                      |

### oplot.util.cast_inputs(\*args_cast, \*\*kwargs_cast)

Creates a decorator that casts the arguments and keyword arguments of a function
according to specified casting functions.

### Parameters

\*\*kwargs_cast
: Mapping of keyword argument names to casting functions.

### Returns

### oplot.util.fixed_step_chunker(it, chk_size, chk_step=None, start_at=None, stop_at=None, return_tail=False)

a function to get (an iterator of) segments (bt, tt) of chunks from an iterator (or list)
of the for [it_1, it_2…], given a chk_size, chk_step, and a start_at and a stop_at.
The start_at, stop_at act like slices indices for a list: start_at is included and stop_at is excluded

* **Parameters:**
  * **it** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)) – iterator of elements of any type
  * **chk_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – length of the chunks
  * **chk_step** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – step between chunks
  * **start_at** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – index of the first term of the iterator at which we begin building the chunks (inclusive)
  * **stop_at** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – index of the last term from the iterator included in the chunks
  * **return_tail** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – if set to false, only the chunks with max element with index less than stop_at are yielded
    if set to true, any chunks with minimum index value no more than stop_at are returned but they contain
    term with index no more than stop_at
* **Returns:**
  an iterator of the chunks

1. If stop_at is not None and return_tail is False:
   will return all full chunks with maximum element index less than stop_at
   or until the iterator is exhausted. Only full chunks are returned here.
2. If stop_at is not None and return_tail is True:
   will return all full chunks as above along with possibly cut off chunks
   containing one term whose index is stop_at-1 or one (last) term which is the
   last element of it
3. If stop_at is None and return_tail is False:
   will return all full chunks with maximum element index less or equal to the last
   element of it
4. If stop_at is None and return_tail is True:
   will return all full chunks with maximum element index less or equal to the last
   element of it plus cut off chunks whose maximum term index is the last term of it

> ### Examples

> ```pycon
> >>> list(fixed_step_chunker([1,2,3,4,5,6,7], chk_size=3, chk_step=2))
> [[1, 2, 3], [3, 4, 5], [5, 6, 7]]
> ```

> Testing chk_step < chk_size with return_tail=TRUE, stop and start_at PRESENT
> and stop_at SMALLER than the largest index of it

> ```pycon
> >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=1, start_at=2, stop_at=5, return_tail=True)
> >>> it = list(range(1, 17, 1))
> >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
> >>> assert A == B  # it and iter(it) should give the same thing!
> >>> A  # and that thing is:
> [[3, 4, 5], [4, 5], [5]]
> ```

> Testing chk_step < chk_size with return_tail=FALSE, stop and start_at PRESENT
> and stop_at SMALLER than the largest index of it

> ```pycon
> >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=1, start_at=2, stop_at=5, return_tail=False)
> >>> it = list(range(1, 17, 1))
> >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
> >>> assert A == B  # it and iter(it) should give the same thing!
> >>> A  # and that thing is:
> [[3, 4, 5]]
> ```

> Testing chk_step < chk_size with return_tail=TRUE, stop and start_at PRESENT
> and stop_at LARGER than the largest index of it

> ```pycon
> >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=1, start_at=1, stop_at=20, return_tail=True)
> >>> it = list(range(1, 17, 1))
> >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
> >>> assert A == B  # it and iter(it) should give the same thing!
> >>> A  # and that thing is:
> [[2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12], [11, 12, 13], [12, 13, 14], [13, 14, 15], [14, 15, 16], [15, 16], [16]]
> ```

> Testing chk_step < chk_size with return_tail=FALSE, stop and start_at PRESENT
> and stop_at LARGER than the largest index of it

> ```pycon
> >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=1, start_at=1, stop_at=20, return_tail=False)
> >>> it = list(range(1, 17, 1))
> >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
> >>> assert A == B  # it and iter(it) should give the same thing!
> >>> A  # and that thing is:
> [[2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12], [11, 12, 13], [12, 13, 14], [13, 14, 15], [14, 15, 16]]
> ```

> Testing chk_step = chk_size with return_tail=TRUE, stop and start_at PRESENT
> and stop_at SMALLER than the largest index of it

> ```pycon
> >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=3, start_at=1, stop_at=7, return_tail=True)
> >>> it = list(range(1, 17, 1))
> >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
> >>> assert A == B  # it and iter(it) should give the same thing!
> >>> A  # and that thing is:
> [[2, 3, 4], [5, 6, 7]]
> ```

> Testing chk_size > len(it) with return_tail=False, no stop_at or start_at

> ```pycon
> >>> f = lambda it: fixed_step_chunker(it, chk_size=30, chk_step=3, start_at=None, stop_at=None, return_tail=False)
> >>> it = list(range(1, 17, 1))
> >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
> >>> assert A == B  # it and iter(it) should give the same thing!
> >>> A  # and that thing is:
> []
> ```

> Testing chk_size > len(it) with return_tail=True, no stop_at or start_at

> ```pycon
> >>> f = lambda it: fixed_step_chunker(it, chk_size=30, chk_step=3, start_at=None, stop_at=None, return_tail=True)
> >>> it = list(range(1, 17, 1))
> >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
> >>> assert A == B  # it and iter(it) should give the same thing!
> >>> A  # and that thing is:
> [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [10, 11, 12, 13, 14, 15, 16], [13, 14, 15, 16], [16]]
> ```

> Testing chk_step > chk_size with return_tail=TRUE, stop and start_at PRESENT
> and stop_at SMALLER than the largest index of it

> ```pycon
> >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=4, start_at=1, stop_at=7, return_tail=True)
> >>> it = list(range(1, 17, 1))
> >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
> >>> assert A == B  # it and iter(it) should give the same thing!
> >>> A  # and that thing is:
> [[2, 3, 4], [6, 7]]
> ```

> Testing chk_step > chk_size with return_tail=FALSE, stop and start_at PRESENT
> and stop_at SMALLER than the largest index of it

> ```pycon
> >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=4, start_at=1, stop_at=7, return_tail=False)
> >>> it = list(range(1, 17, 1))
> >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
> >>> assert A == B  # it and iter(it) should give the same thing!
> >>> A  # and that thing is:
> [[2, 3, 4]]
> ```

> Testing chk_step > chk_size with return_tail=FALSE, stop and start_at NOT PRESENT

> ```pycon
> >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=4, start_at=None, stop_at=None, return_tail=False)
> >>> it = list(range(1, 17, 1))
> >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
> >>> assert A == B  # it and iter(it) should give the same thing!
> >>> A  # and that thing is:
> [[1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15]]
> ```

> Testing chk_step > chk_size with return_tail=TRUE, stop and start_at NOT PRESENT

> ```pycon
> >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=4, start_at=None, stop_at=None, return_tail=True)
> >>> it = list(range(1, 19, 1))
> >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
> >>> assert A == B  # it and iter(it) should give the same thing!
> >>> A  # and that thing is:
> [[1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15], [17, 18]]
> ```

> Testing chk_step > chk_size with return_tail=TRUE, stop and start_at NOT PRESENT
> with negative values in the iterator

> ```pycon
> >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=4, start_at=None, stop_at=None, return_tail=True)
> >>> it = list(range(-10, 19, 1))
> >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
> >>> assert A == B  # it and iter(it) should give the same thing!
> >>> A  # and that thing is:
> [[-10, -9, -8], [-6, -5, -4], [-2, -1, 0], [2, 3, 4], [6, 7, 8], [10, 11, 12], [14, 15, 16], [18]]
> ```

> Testing chk_step > chk_size with return_tail=TRUE, stop and start_at NOT PRESENT
> with items of various types in the iterator

> ```pycon
> >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=2, start_at=None, stop_at=None, return_tail=True)
> >>> it = ['a', 3, -10, 9.2, str, [1,2,3], set([10,20])]
> >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
> >>> assert A == B  # it and iter(it) should give the same thing!
> >>> A  # and that thing is:
> [['a', 3, -10], [-10, 9.2, <class 'str'>], [<class 'str'>, [1, 2, 3], {10, 20}], [{10, 20}]]
> ```

### oplot.util.float_to_timestamp(floats, reference_time=Timestamp('1970-01-01 00:00:00'))

Convert an array-like of floats (days since reference datetime) back to Timestamps.

### Parameters

reference_time
: The fixed reference datetime (default: ‘1970-01-01’).

### Returns

### oplot.util.timestamp_to_float(timestamps, reference_time=Timestamp('1970-01-01 00:00:00'))

Convert an array-like of Timestamps to absolute floats representing days
since a fixed reference datetime.

### Parameters

reference_time
: The fixed reference datetime (default: ‘1970-01-01’).

### Returns
