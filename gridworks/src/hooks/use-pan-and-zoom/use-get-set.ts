/**
 * Code from https://github.com/wouterraateland/use-pan-and-zoom.
 *
 * MIT License
 *
 * Copyright (c) 2019 Wouter Raateland
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
import { useCallback, useRef } from "react";

export function useGetSet<T>(
  initialValue: T
): [() => T, (value: ((current: T) => T) | T) => T] {
  const ref = useRef(initialValue);
  const get = useCallback(() => ref.current, []);
  const set = useCallback((value: ((current: T) => T) | T) => {
    if (typeof value === "function") {
      ref.current = (value as (current: T) => T)(ref.current);
    } else {
      ref.current = value;
    }
    return ref.current;
  }, []);

  return [get, set];
}
