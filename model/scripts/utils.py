import random
import time
from skimage import morphology
import matplotlib.pyplot as plt
import math


#####################################################################
# Jitendra-Sampling相关
#####################################################################

################################################
# 快排
################################################
def partition(li, left, right):
    tmp = li[left]
    while left < right:
        while left < right and li[right] >= tmp:  # 从右边找比tmp小的数
            right -= 1  # 继续从右往左查找
        li[left] = li[right]  # 把右边的值写到左边空位上

        while left < right and li[left] <= tmp:
            left += 1
        li[right] = li[left]  # 把左边的值写到右边空位上

    li[left] = tmp  # 把tmp归位
    return left


def quick_sort(li, left, right):
    if left < right:  # 至少两个元素
        mid = partition(li, left, right)
        quick_sort(li, left, mid - 1)
        quick_sort(li, mid + 1, right)
        return True
    else:
        return False


################################################
# Jitendra’s Sampling采样(cv2)
################################################
def Jitendra(points, final_num, thresh=3):
    # print('计时开始：')
    # begTime = time.perf_counter()

    # 若轮廓点数量过大，则只取thresh*final_num以减小复杂度
    if len(points) > thresh*final_num:
        random.shuffle(points)
        points = points[:thresh*final_num]

    length = len(points)
    distance = []
    for i in range(length):
        for j in range(length):
            if i !=j:
                dist = math.sqrt(pow(points[i][0] - points[j][0], 2) + pow(points[i][1] - points[j][1], 2))
                distance.append({'index1' : i, 'index2' : j, 'dist' : dist})

    # node1Time = time.perf_counter()
    #　print('计时结点1-完成距离计算：' + str(node1Time - begTime) + '秒')

    distance.sort(key=lambda x: x['dist'])   # 用封装好的函数根据距离进行排序
    # print(re)

    # node2Time = time.perf_counter()
    # print('计时结点2-完成快排：' + str(node2Time - begTime) + '秒')

    need_remove_num = length - final_num
    removed = np.zeros((length), dtype = np.int8)
    i = 0
    while need_remove_num > 0:
        if removed[distance[i]['index1']] == 0 and removed[distance[i]['index2']] == 0:
            removed[distance[i]['index1']] = 1
            need_remove_num -= 1
        i += 1

    re = []
    for i in range(length):
        if removed[i] == 0:
            re.append(points[i])

    # node3Time = time.perf_counter()
    # print('计时结点3-完成挑选：' + str(node3Time - begTime) + '秒')

    # endTime = time.perf_counter()
    # print('计时结束：' + str(endTime - begTime) + '秒')
    return re


################################################
# 提取轮廓
################################################
def get_contour(img):
    # get contour
    contour_img = np.zeros(shape=img.shape, dtype=np.uint8)
    contour_list = []
    contour_img += 255
    h = img.shape[0]
    w = img.shape[1]
    for i in range(1, h-1):
        for j in range(1, w-1):
            if img[i][j] == 0:
                contour_img[i][j] = 0
                sum = 0
                sum += img[i - 1][j + 1]
                sum += img[i][j + 1]
                sum += img[i + 1][j + 1]
                sum += img[i - 1][j]
                sum += img[i + 1][j]
                sum += img[i - 1][j - 1]
                sum += img[i][j - 1]
                sum += img[i + 1][j - 1]
                if sum == 0:
                    contour_img[i][j] = 255
                else:
                    contour_list.append([i, j])
    return contour_img, contour_list


################################################
# 求字形轮廓点
################################################
def get_contour_sample_points(image, num, show=False):
    _, points = get_contour(image)
    re = Jitendra(points, num)  # 提取轮廓点

    if show:
        # 简单验证一下处理结果
        finalimg = np.zeros((256, 256), np.uint8)
        for item in re:
            finalimg[item[0]][item[1]] = 255

        plt.set_cmap('binary')
        plt.rcParams['font.sans-serif'] = ['KaiTi']
        plt.rcParams['axes.unicode_minus'] = False
        plt.imshow(finalimg)
        plt.title('Contour_points_based_on_Jitendra_Sampling', fontsize=16)
        plt.show()

    return re


################################################
# 求骨架散点
################################################
def get_ske_sample_points(image, num, show=False):
    tmp = 1 - image.astype(np.uint8) / 255  # skeletonize无法直接识别cv库图片，需转换
    ske = morphology.skeletonize(tmp)  # 细化
    points = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if ske[x][y] == 1:  # 获得的ske图像中白点(1)是前景点
                points.append([x, y])
    re = Jitendra(points, num)  # 提取轮廓点

    if show:
        # 简单验证一下处理结果
        finalimg = np.zeros((256, 256, 1), np.uint8)
        for item in re:
            finalimg[item[0]][item[1]] = 255

        plt.set_cmap('binary')
        plt.rcParams['font.sans-serif'] = ['KaiTi']
        plt.rcParams['axes.unicode_minus'] = False
        plt.imshow(finalimg)
        plt.title('skeleton_points_based_on_Jitendra_Sampling', fontsize=16)
        plt.show()

    return re



#####################################################################
# Shape-Context相关
#####################################################################
"""
Introduction
============

The Munkres module provides an implementation of the Munkres algorithm
(also called the Hungarian algorithm or the Kuhn-Munkres algorithm),
useful for solving the Assignment Problem.

Assignment Problem
==================

Let *C* be an *n*\ x\ *n* matrix representing the costs of each of *n* workers
to perform any of *n* jobs. The assignment problem is to assign jobs to
workers in a way that minimizes the total cost. Since each worker can perform
only one job and each job can be assigned to only one worker the assignments
represent an independent set of the matrix *C*.

One way to generate the optimal set is to create all permutations of
the indexes necessary to traverse the matrix so that no row and column
are used more than once. For instance, given this matrix (expressed in
Python)::

    matrix = [[5, 9, 1],
              [10, 3, 2],
              [8, 7, 4]]

You could use this code to generate the traversal indexes::

    def permute(a, results):
        if len(a) == 1:
            results.insert(len(results), a)

        else:
            for i in range(0, len(a)):
                element = a[i]
                a_copy = [a[j] for j in range(0, len(a)) if j != i]
                subresults = []
                permute(a_copy, subresults)
                for subresult in subresults:
                    result = [element] + subresult
                    results.insert(len(results), result)

    results = []
    permute(range(len(matrix)), results) # [0, 1, 2] for a 3x3 matrix

After the call to permute(), the results matrix would look like this::

    [[0, 1, 2],
     [0, 2, 1],
     [1, 0, 2],
     [1, 2, 0],
     [2, 0, 1],
     [2, 1, 0]]

You could then use that index matrix to loop over the original cost matrix
and calculate the smallest cost of the combinations::

    n = len(matrix)
    minval = sys.maxsize
    for row in range(n):
        cost = 0
        for col in range(n):
            cost += matrix[row][col]
        minval = min(cost, minval)

    print minval

While this approach works fine for small matrices, it does not scale. It
executes in O(*n*!) time: Calculating the permutations for an *n*\ x\ *n*
matrix requires *n*! operations. For a 12x12 matrix, that's 479,001,600
traversals. Even if you could manage to perform each traversal in just one
millisecond, it would still take more than 133 hours to perform the entire
traversal. A 20x20 matrix would take 2,432,902,008,176,640,000 operations. At
an optimistic millisecond per operation, that's more than 77 million years.

The Munkres algorithm runs in O(*n*\ ^3) time, rather than O(*n*!). This
package provides an implementation of that algorithm.

This version is based on
http://www.public.iastate.edu/~ddoty/HungarianAlgorithm.html.

This version was written for Python by Brian Clapper from the (Ada) algorithm
at the above web site. (The ``Algorithm::Munkres`` Perl version, in CPAN, was
clearly adapted from the same web site.)

Usage
=====

Construct a Munkres object::

    from munkres import Munkres

    m = Munkres()

Then use it to compute the lowest cost assignment from a cost matrix. Here's
a sample program::

    from munkres import Munkres, print_matrix

    matrix = [[5, 9, 1],
              [10, 3, 2],
              [8, 7, 4]]
    m = Munkres()
    indexes = m.compute(matrix)
    print_matrix(matrix, msg='Lowest cost through this matrix:')
    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
        print '(%d, %d) -> %d' % (row, column, value)
    print 'total cost: %d' % total

Running that program produces::

    Lowest cost through this matrix:
    [5, 9, 1]
    [10, 3, 2]
    [8, 7, 4]
    (0, 0) -> 5
    (1, 1) -> 3
    (2, 2) -> 4
    total cost=12

The instantiated Munkres object can be used multiple times on different
matrices.

Non-square Cost Matrices
========================

The Munkres algorithm assumes that the cost matrix is square. However, it's
possible to use a rectangular matrix if you first pad it with 0 values to make
it square. This module automatically pads rectangular cost matrices to make
them square.

Notes:

- The module operates on a *copy* of the caller's matrix, so any padding will
  not be seen by the caller.
- The cost matrix must be rectangular or square. An irregular matrix will
  *not* work.

Calculating Profit, Rather than Cost
====================================

The cost matrix is just that: A cost matrix. The Munkres algorithm finds
the combination of elements (one from each row and column) that results in
the smallest cost. It's also possible to use the algorithm to maximize
profit. To do that, however, you have to convert your profit matrix to a
cost matrix. The simplest way to do that is to subtract all elements from a
large value. For example::

    from munkres import Munkres, print_matrix

    matrix = [[5, 9, 1],
              [10, 3, 2],
              [8, 7, 4]]
    cost_matrix = []
    for row in matrix:
        cost_row = []
        for col in row:
            cost_row += [sys.maxsize - col]
        cost_matrix += [cost_row]

    m = Munkres()
    indexes = m.compute(cost_matrix)
    print_matrix(matrix, msg='Highest profit through this matrix:')
    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
        print '(%d, %d) -> %d' % (row, column, value)

    print 'total profit=%d' % total

Running that program produces::

    Highest profit through this matrix:
    [5, 9, 1]
    [10, 3, 2]
    [8, 7, 4]
    (0, 1) -> 9
    (1, 0) -> 10
    (2, 2) -> 4
    total profit=23

The ``munkres`` module provides a convenience method for creating a cost
matrix from a profit matrix. Since it doesn't know whether the matrix contains
floating point numbers, decimals, or integers, you have to provide the
conversion function; but the convenience method takes care of the actual
creation of the cost matrix::

    import munkres

    cost_matrix = munkres.make_cost_matrix(matrix,
                                           lambda cost: sys.maxsize - cost)

So, the above profit-calculation program can be recast as::

    from munkres import Munkres, print_matrix, make_cost_matrix

    matrix = [[5, 9, 1],
              [10, 3, 2],
              [8, 7, 4]]
    cost_matrix = make_cost_matrix(matrix, lambda cost: sys.maxsize - cost)
    m = Munkres()
    indexes = m.compute(cost_matrix)
    print_matrix(matrix, msg='Lowest cost through this matrix:')
    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
        print '(%d, %d) -> %d' % (row, column, value)
    print 'total profit=%d' % total

References
==========

1. http://www.public.iastate.edu/~ddoty/HungarianAlgorithm.html

2. Harold W. Kuhn. The Hungarian Method for the assignment problem.
   *Naval Research Logistics Quarterly*, 2:83-97, 1955.

3. Harold W. Kuhn. Variants of the Hungarian method for assignment
   problems. *Naval Research Logistics Quarterly*, 3: 253-258, 1956.

4. Munkres, J. Algorithms for the Assignment and Transportation Problems.
   *Journal of the Society of Industrial and Applied Mathematics*,
   5(1):32-38, March, 1957.

5. http://en.wikipedia.org/wiki/Hungarian_algorithm

Copyright and License
=====================

This software is released under a BSD license, adapted from
<http://opensource.org/licenses/bsd-license.php>

Copyright (c) 2008 Brian M. Clapper
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name "clapper.org" nor the names of its contributors may be
  used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
__docformat__ = 'restructuredtext'
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import sys
import copy

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
__all__     = ['Munkres', 'make_cost_matrix']

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
# Info about the module
__version__   = "1.0.6"
__author__    = "Brian Clapper, bmc@clapper.org"
__url__       = "http://software.clapper.org/munkres/"
__copyright__ = "(c) 2008 Brian M. Clapper"
__license__   = "BSD-style license"

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
class Munkres:
    """
    Calculate the Munkres solution to the classical assignment problem.
    See the module documentation for usage.
    """

    def __init__(self):
        """Create a new instance"""
        self.C = None
        self.row_covered = []
        self.col_covered = []
        self.n = 0
        self.Z0_r = 0
        self.Z0_c = 0
        self.marked = None
        self.path = None

    def make_cost_matrix(profit_matrix, inversion_function):
        """
        **DEPRECATED**

        Please use the module function ``make_cost_matrix()``.
        """
        import munkres
        return munkres.make_cost_matrix(profit_matrix, inversion_function)

    make_cost_matrix = staticmethod(make_cost_matrix)

    def pad_matrix(self, matrix, pad_value=0):
        """
        Pad a possibly non-square matrix to make it square.

        :Parameters:
            matrix : list of lists
                matrix to pad

            pad_value : int
                value to use to pad the matrix

        :rtype: list of lists
        :return: a new, possibly padded, matrix
        """
        max_columns = 0
        total_rows = len(matrix)

        for row in matrix:
            max_columns = max(max_columns, len(row))

        total_rows = max(max_columns, total_rows)

        new_matrix = []
        for row in matrix:
            row_len = len(row)
            new_row = row[:]
            if total_rows > row_len:
                # Row too short. Pad it.
                new_row += [0] * (total_rows - row_len)
            new_matrix += [new_row]

        while len(new_matrix) < total_rows:
            new_matrix += [[0] * total_rows]

        return new_matrix

    def compute(self, cost_matrix):
        """
        Compute the indexes for the lowest-cost pairings between rows and
        columns in the database. Returns a list of (row, column) tuples
        that can be used to traverse the matrix.

        :Parameters:
            cost_matrix : list of lists
                The cost matrix. If this cost matrix is not square, it
                will be padded with zeros, via a call to ``pad_matrix()``.
                (This method does *not* modify the caller's matrix. It
                operates on a copy of the matrix.)

                **WARNING**: This code handles square and rectangular
                matrices. It does *not* handle irregular matrices.

        :rtype: list
        :return: A list of ``(row, column)`` tuples that describe the lowest
                 cost path through the matrix

        """
        self.C = self.pad_matrix(cost_matrix)
        self.n = len(self.C)
        self.original_length = len(cost_matrix)
        self.original_width = len(cost_matrix[0])
        self.row_covered = [False for i in range(self.n)]
        self.col_covered = [False for i in range(self.n)]
        self.Z0_r = 0
        self.Z0_c = 0
        self.path = self.__make_matrix(self.n * 2, 0)
        self.marked = self.__make_matrix(self.n, 0)

        done = False
        step = 1

        steps = { 1 : self.__step1,
                  2 : self.__step2,
                  3 : self.__step3,
                  4 : self.__step4,
                  5 : self.__step5,
                  6 : self.__step6 }

        while not done:
            try:
                func = steps[step]
                step = func()
            except KeyError:
                done = True

        # Look for the starred columns
        results = []
        for i in range(self.original_length):
            for j in range(self.original_width):
                if self.marked[i][j] == 1:
                    results += [(i, j)]

        return results

    def __copy_matrix(self, matrix):
        """Return an exact copy of the supplied matrix"""
        return copy.deepcopy(matrix)

    def __make_matrix(self, n, val):
        """Create an *n*x*n* matrix, populating it with the specific value."""
        matrix = []
        for i in range(n):
            matrix += [[val for j in range(n)]]
        return matrix

    def __step1(self):
        """
        For each row of the matrix, find the smallest element and
        subtract it from every element in its row. Go to Step 2.
        """
        C = self.C
        n = self.n
        for i in range(n):
            minval = min(self.C[i])
            # Find the minimum value for this row and subtract that minimum
            # from every element in the row.
            for j in range(n):
                self.C[i][j] -= minval

        return 2

    def __step2(self):
        """
        Find a zero (Z) in the resulting matrix. If there is no starred
        zero in its row or column, star Z. Repeat for each element in the
        matrix. Go to Step 3.
        """
        n = self.n
        for i in range(n):
            for j in range(n):
                if (self.C[i][j] == 0) and \
                        (not self.col_covered[j]) and \
                        (not self.row_covered[i]):
                    self.marked[i][j] = 1
                    self.col_covered[j] = True
                    self.row_covered[i] = True

        self.__clear_covers()
        return 3

    def __step3(self):
        """
        Cover each column containing a starred zero. If K columns are
        covered, the starred zeros describe a complete set of unique
        assignments. In this case, Go to DONE, otherwise, Go to Step 4.
        """
        n = self.n
        count = 0
        for i in range(n):
            for j in range(n):
                if self.marked[i][j] == 1:
                    self.col_covered[j] = True
                    count += 1

        if count >= n:
            step = 7 # done
        else:
            step = 4

        return step

    def __step4(self):
        """
        Find a noncovered zero and prime it. If there is no starred zero
        in the row containing this primed zero, Go to Step 5. Otherwise,
        cover this row and uncover the column containing the starred
        zero. Continue in this manner until there are no uncovered zeros
        left. Save the smallest uncovered value and Go to Step 6.
        """
        step = 0
        done = False
        row = -1
        col = -1
        star_col = -1
        while not done:
            (row, col) = self.__find_a_zero()
            if row < 0:
                done = True
                step = 6
            else:
                self.marked[row][col] = 2
                star_col = self.__find_star_in_row(row)
                if star_col >= 0:
                    col = star_col
                    self.row_covered[row] = True
                    self.col_covered[col] = False
                else:
                    done = True
                    self.Z0_r = row
                    self.Z0_c = col
                    step = 5

        return step

    def __step5(self):
        """
        Construct a series of alternating primed and starred zeros as
        follows. Let Z0 represent the uncovered primed zero found in Step 4.
        Let Z1 denote the starred zero in the column of Z0 (if any).
        Let Z2 denote the primed zero in the row of Z1 (there will always
        be one). Continue until the series terminates at a primed zero
        that has no starred zero in its column. Unstar each starred zero
        of the series, star each primed zero of the series, erase all
        primes and uncover every line in the matrix. Return to Step 3
        """
        count = 0
        path = self.path
        path[count][0] = self.Z0_r
        path[count][1] = self.Z0_c
        done = False
        while not done:
            row = self.__find_star_in_col(path[count][1])
            if row >= 0:
                count += 1
                path[count][0] = row
                path[count][1] = path[count-1][1]
            else:
                done = True

            if not done:
                col = self.__find_prime_in_row(path[count][0])
                count += 1
                path[count][0] = path[count-1][0]
                path[count][1] = col

        self.__convert_path(path, count)
        self.__clear_covers()
        self.__erase_primes()
        return 3

    def __step6(self):
        """
        Add the value found in Step 4 to every element of each covered
        row, and subtract it from every element of each uncovered column.
        Return to Step 4 without altering any stars, primes, or covered
        lines.
        """
        minval = self.__find_smallest()
        for i in range(self.n):
            for j in range(self.n):
                if self.row_covered[i]:
                    self.C[i][j] += minval
                if not self.col_covered[j]:
                    self.C[i][j] -= minval
        return 4

    def __find_smallest(self):
        """Find the smallest uncovered value in the matrix."""
        minval = sys.maxsize
        for i in range(self.n):
            for j in range(self.n):
                if (not self.row_covered[i]) and (not self.col_covered[j]):
                    if minval > self.C[i][j]:
                        minval = self.C[i][j]
        return minval

    def __find_a_zero(self):
        """Find the first uncovered element with value 0"""
        row = -1
        col = -1
        i = 0
        n = self.n
        done = False

        while not done:
            j = 0
            while True:
                if (self.C[i][j] == 0) and \
                        (not self.row_covered[i]) and \
                        (not self.col_covered[j]):
                    row = i
                    col = j
                    done = True
                j += 1
                if j >= n:
                    break
            i += 1
            if i >= n:
                done = True

        return (row, col)

    def __find_star_in_row(self, row):
        """
        Find the first starred element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
        col = -1
        for j in range(self.n):
            if self.marked[row][j] == 1:
                col = j
                break

        return col

    def __find_star_in_col(self, col):
        """
        Find the first starred element in the specified row. Returns
        the row index, or -1 if no starred element was found.
        """
        row = -1
        for i in range(self.n):
            if self.marked[i][col] == 1:
                row = i
                break

        return row

    def __find_prime_in_row(self, row):
        """
        Find the first prime element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
        col = -1
        for j in range(self.n):
            if self.marked[row][j] == 2:
                col = j
                break

        return col

    def __convert_path(self, path, count):
        for i in range(count+1):
            if self.marked[path[i][0]][path[i][1]] == 1:
                self.marked[path[i][0]][path[i][1]] = 0
            else:
                self.marked[path[i][0]][path[i][1]] = 1

    def __clear_covers(self):
        """Clear all covered matrix cells"""
        for i in range(self.n):
            self.row_covered[i] = False
            self.col_covered[i] = False

    def __erase_primes(self):
        """Erase all prime markings"""
        for i in range(self.n):
            for j in range(self.n):
                if self.marked[i][j] == 2:
                    self.marked[i][j] = 0

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
def make_cost_matrix(profit_matrix, inversion_function):
    """
    Create a cost matrix from a profit matrix by calling
    'inversion_function' to invert each value. The inversion
    function must take one numeric argument (of any type) and return
    another numeric argument which is presumed to be the cost inverse
    of the original profit.

    This is a static method. Call it like this:

    .. python::

        cost_matrix = Munkres.make_cost_matrix(matrix, inversion_func)

    For example:

    .. python::

        cost_matrix = Munkres.make_cost_matrix(matrix, lambda x : sys.maxsize - x)

    :Parameters:
        profit_matrix : list of lists
            The matrix to convert from a profit to a cost matrix

        inversion_function : function
            The function to use to invert each entry in the profit matrix

    :rtype: list of lists
    :return: The converted matrix
    """
    cost_matrix = []
    for row in profit_matrix:
        cost_matrix.append([inversion_function(value) for value in row])
    return cost_matrix

def print_matrix(matrix, msg=None):
    """
    Convenience function: Displays the contents of a matrix of integers.

    :Parameters:
        matrix : list of lists
            Matrix to print

        msg : str
            Optional message to print before displaying the matrix
    """
    import math

    if msg is not None:
        print(msg)

    # Calculate the appropriate format width.
    width = 0
    for row in matrix:
        for val in row:
            width = max(width, int(math.log10(val)) + 1)

    # Make the format string
    format = '%%%dd' % width

    # Print the matrix
    for row in matrix:
        sep = '['
        for val in row:
            sys.stdout.write(sep + format % val)
            sep = ', '
        sys.stdout.write(']\n')


#####################################################################
# 其他工具函数
#####################################################################
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image



#####################################################################
# 工具函数
#####################################################################
import os

################################################
# 删除一个文件夹下的所有所有文件
################################################
def del_files(filepath):
    ls = os.listdir(filepath)
    for i in ls:
        c_path = os.path.join(filepath, i)
        if os.path.isdir(c_path):  # 如果是文件夹那么递归调用一下
            del_files(c_path)
        else:  # 如果是一个文件那么直接删除
            os.remove(c_path)
    print('文件夹清空完成')


################################################
# 二值化(cv2)
################################################
# 不要再使用imread的默认参数GRAYSCALE了！https://blog.csdn.net/zhaoxi_li/article/details/102529160 #
def ostu(img0, thresh1=0):
    gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)  # 灰度化
    box = cv.boxFilter(gray, -1, (3, 3), normalize=True)  # 去噪
    _, binarized = cv.threshold(box, thresh1, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 二值化
    return binarized


################################################
# cv2转PIL
################################################
def cv2_to_PIL(img):
    img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    return img


################################################
# PIL转cv2
################################################
def PIL_to_cv2(img):
    img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
    return img


################################################
# 求重心(cv2)
################################################
def G(img):
    h = img.shape[0]
    w = img.shape[1]
    xfenzi = 0
    yfenzi = 0
    fenmu = 0
    for x in range(h):
        for y in range(w):
            fenmu += img[x, y] / 255
            xfenzi += x * img[x, y] / 255
            yfenzi += y * img[x, y] / 255
    fx = xfenzi / fenmu
    fy = yfenzi / fenmu
    # print('重心：[' + str(fx) + ',' + str(fy) + ']')
    return fx, fy


################################################
# 求点集中心（cv2）
################################################
def G_points(img,points):
    xfenzi = 0
    yfenzi = 0
    fenmu = 0
    for [x,y] in points:
        fenmu += 1
        xfenzi += x
        yfenzi += y
    fx = xfenzi / fenmu
    fy = yfenzi / fenmu
    # print('重心：[' + str(fx) + ',' + str(fy) + ']')
    return fx, fy


# import math
# from scripts.projection import get_vert_proj, get_hori_proj

# """"
# cnt1=cnt2=0
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         if img[i, j]==0:
#             cnt1 += 1
# for i in range(img2.shape[0]):
#     for j in range(img2.shape[1]):
#         if img2[i, j]==0:
#             cnt2 += 1
# print(cnt1,cnt2)
# """


# ################################################
# # 压力变化特征(cv2)
# ################################################
# def stress(img, fx, fy):
#     h = img.shape[0]
#     w = img.shape[1]
#     xfenzi = 0
#     yfenzi = 0
#     xfenmu = 0
#     yfenmu = 0
#     for x in range(h):
#         for y in range(w):
#             xfenzi += pow(x-fx,3) * img[x, y]

#             xfenmu += pow(x-fx,3) * img[x, y]
#             if x <= math.floor(h/2)-1:
#                 xfenmu += pow(x-fx,3) * img[x, y]

#             if x>=math.floor(h/2) and y>=math.floor(w/2):
#                 yfenzi += pow(x-fy,3) * img[x, y]

#             yfenmu += pow(y - fy, 3) * img[x, y]
#             if y <= math.floor(w/2)-1:
#                 yfenmu += pow(x-fy,3) * img[x, y]

#     stress_x = xfenzi/xfenmu
#     stress_y = yfenzi/yfenmu
#     # print('压力变化特征：[' + str(stress_x) + ',' + str(stress_y) + ']')
#     return stress_x, stress_y


# ################################################
# # 倾斜平衡特征(cv2)
# ################################################
# def slant(img, fx, fy):
#     h = img.shape[0]
#     w = img.shape[1]
#     xfenzi = 0
#     yfenzi = 0
#     xfenmu = 0
#     yfenmu = 0
#     for x in range(h):
#         for y in range(w):
#             if x >= math.floor(h / 2):
#                 yfenzi += (x - fx) * (x - fx) * (y - fy) * img[x, y]

#             yfenmu += (x - fx) * (x - fx) * (y - fy) * img[x, y]
#             if x <= math.floor(h / 2):
#                 yfenmu += (x - fy) * (x - fy) * (y - fy) * img[x, y]

#             if y >= math.floor(w / 2):
#                 xfenzi += (x - fx) * (y - fy) * img[x, y]

#             if y >= math.floor(w / 2):
#                 xfenmu += (x - fx) * (y - fy) * img[x, y]
#             if y <= math.floor(w / 2) - 1:
#                 xfenmu += (x - fx) * (y - fy) * (y - fy) * img[x, y]

#     slant_x = xfenzi / xfenmu
#     slant_y = yfenzi / yfenmu
#     # print('倾斜平衡特征：[' + str(slant_x) + ',' + str(slant_y) + ']')
#     return slant_x, slant_y


# ################################################
# # 字高宽比(PIL)
# ################################################
# def h_w_rate(img):
#     points1, _ = get_hori_proj(img)
#     points2, _ = get_vert_proj(img)
#     tmp1 = 0
#     tmp2=len(points1)
#     for i in range(len(points1) - 1):
#         if points1[i] == 0 and points1[i + 1] != 0:
#             tmp1 = i
#         if points1[i + 1] == 0 and points1[i] != 0:
#             tmp2 = i
#     c_height = tmp2 - tmp1
#     tmp1 = 0
#     tmp2 = len(points2)
#     for i in range(len(points2) - 1):
#         if points2[i] == 0 and points2[i + 1] != 0:
#             tmp1 = i
#         if points2[i + 1] == 0 and points2[i] != 0:
#             tmp2 = i
#     c_width = tmp2 - tmp1
#     return c_height / c_width


# def fg_points_percent(img, centerx, centery, r):
#     fg_cnt = 0  # 前景点个数
#     total_cnt = 0     # 圆内点个数
#     for i in range(centerx-r, centerx+r+1):
#         for j in range(centery-r,centery+r+1):
#             if i < img.shape[0] and j < img.shape[1]:
#                 if pow(i-centerx, 2) + pow(j-centery, 2) < pow(r, 2):
#                     total_cnt += 1
#                     if img[i, j]==0:
#                         fg_cnt += 1
#     return fg_cnt/total_cnt


# ################################################
# # 平均笔画宽度(cv2)
# ################################################
# def mean_stroke_width(img):
#     ske_points = get_ske_sample_points(img, 100)
#     d=[]
#     for points in ske_points:
#         r=1
#         while True:
#             if fg_points_percent(img, points[0], points[1], r) < 0.95:
#                 break
#             r += 1
#         d.append(r)
#     return sum(d)/len(d)


# # 字体大小不定，不好用啊...
# def fun(img1, img2):
#     contour1 = get_contour_sample_points(img1, 150)
#     contour2 = get_contour_sample_points(img2, 150)
#     ske1 = get_ske_sample_points(img1, 80)
#     ske2 = get_ske_sample_points(img2, 80)
#     gx1, gy1 = G(img1)
#     gx2, gy2 = G(img2)

#     # 尝试构造一个距离矩阵（意识到难点是”获取相似度百分比“而不是仅仅获取”差异量“）
#     contour1_cost = []
#     for item in contour1:
#         dist = math.sqrt(pow(item[0] - gx1, 2) + pow(item[1] - gy1, 2))
#         contour1_cost.append(dist)
#     mean1 = sum(contour1_cost) / len(contour1_cost)
#     s1 = 0  # 方差
#     for i in range(len(contour1_cost)):
#         s1 += pow(contour1_cost[i] - mean1, 2)
#         contour1_cost[i] /= mean1
#     s1 /= len(contour1_cost)

#     contour2_cost = []
#     for item in contour2:
#         dist = math.sqrt(pow(item[0] - gx2, 2) + pow(item[1] - gy2, 2))
#         contour2_cost.append(dist)
#     mean2 = sum(contour2_cost) / len(contour2_cost)
#     s2 = 0  # 方差
#     for i in range(len(contour2_cost)):
#         s2 += pow(contour2_cost[i] - mean2, 2)
#         contour2_cost[i] /= mean2
#     s2 /= len(contour2_cost)

#     print('轮廓平均值：',mean1, mean2 )
#     print('轮廓方差：', s1, s2)
#     print('轮廓距离矩阵：')
#     # print(contour1_cost)
#     # print(contour2_cost)

#     # ###
#     ske1_cost = []
#     for item in ske1:
#         dist = math.sqrt(pow(item[0] - gx1, 2) + pow(item[1] - gy1, 2))
#         ske1_cost.append(dist)
#     smean1 = sum(ske1_cost) / len(ske1_cost)
#     ss1 = 0
#     for i in range(len(ske1_cost)):
#         ss1 += pow(ske1_cost[i] - smean1, 2)
#         ske1_cost[i] /= smean1
#     ss1 /= len(ske1_cost)

#     ske2_cost = []
#     for item in ske2:
#         dist = math.sqrt(pow(item[0] - gx1, 2) + pow(item[1] - gy1, 2))
#         ske2_cost.append(dist)
#     smean2 = sum(ske2_cost) / len(ske2_cost)
#     ss2 = 0
#     for i in range(len(ske2_cost)):
#         ss2 += pow(ske2_cost[i] - smean2, 2)
#         ske2_cost[i] /= smean2
#     ss2 /= len(ske2_cost)

#     print('骨架平均值：', smean1, smean2)
#     print('骨架方差：', s1, s2)
#     print('骨架距离矩阵：')
#     # print(ske1_cost)
#     # print(ske2_cost)

#     plt.rcParams['font.sans-serif'] = ['KaiTi']
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.set_cmap('binary')

#     plt.subplot(1, 2, 1)
#     plt.imshow(img1)
#     plt.title("轮廓的平均值u = {:.6f}  {:.6f}\n骨架的平均值u = {:.6f}  {:.6f}".format(mean1, mean2, smean1, smean2), fontsize=14, y=1.2)
#     # plt.title('f4 = {}'.format(f4), y=-0.2)

#     plt.subplot(1, 2, 2)
#     plt.imshow(img2)
#     plt.title("轮廓的方差s = {:.6f}  {:.6f}\n骨架的方差s = {:.6f}  {:.6f}".format(s1, s2, ss1, ss2), fontsize=14, y = -0.4)

#     plt.show()
#     return 1


# def test_slant_stress(img1, img2):
#     gx1, gy1 = G(img1)
#     stressx1, stressy1 = stress(img1, gx1, gy1)
#     slantx1, slanty1 = slant(img1, gx1, gy1)
#     gx2, gy2 = G(img2)
#     stressx2, stressy2 = stress(img2, gx2, gy2)
#     slantx2, slanty2 = slant(img2, gx2, gy2)



#     plt.rcParams['font.sans-serif'] = ['KaiTi']
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.set_cmap('binary')

#     plt.subplot(1, 2, 1)
#     plt.imshow(img1)
#     plt.title("压力变化特征u1 = ({:.6f}, {:.6f})\n压力变化特征u2 = ({:.6f}, {:.6f})".format(stressx1, stressy1, stressx2, stressy2), fontsize=14, y=1.2)
#     # plt.title('f4 = {}'.format(f4), y=-0.2)

#     plt.subplot(1, 2, 2)
#     plt.imshow(img2)
#     plt.title("倾斜平衡特征s1 = ({:.6f}, {:.6f})\n倾斜平衡特征s2 = ({:.6f}, {:.6f})".format(slantx1, slanty1, slantx2, slanty2), fontsize=14, y=-0.4)

#     plt.show()
#     return 0




# """
# ################################################
# # Main函数（要注意绝对值）
# ################################################
# image = cv.imread('1.jpg')
# img = to_cv2_ostu(image)
# gx, gy = G(img)
# print('重心：['+str(gx)+','+str(gy)+']' )
# stress_x, stress_y = stress(img, gx, gy)
# print('压力变化特征：[' + str(stress_x) + ',' + str(stress_y) + ']')
# slant_x, slant_y = slant(img, gx, gy)
# print('倾斜平衡特征：[' + str(slant_x) + ',' + str(slant_y) + ']')

# print('=================================================================')

# image2 = cv.imread('1-1.jpg')
# img2 = to_cv2_ostu(image2)
# gx2, gy2 = G(img2)
# print('重心：['+str(gx2)+','+str(gy2)+']' )
# stress_x2, stress_y2 = stress(img2, gx2, gy2)
# print('压力变化特征：[' + str(stress_x2) + ',' + str(stress_y2) + ']')
# slant_x2, slant_y2 = slant(img2, gx2, gy2)
# print('倾斜平衡特征：[' + str(slant_x2) + ',' + str(slant_y2) + ']')

# # 以1.jpg和1-1.jpg为例，虽然倾斜平衡差异很大你可以这样直接说，但是你还是要评分嘛，所以还是要搞个数据出来
# print('=================================================================')

# image2 = cv.imread('2.jpg')
# img2 = to_cv2_ostu(image2)
# gx2, gy2 = G(img2)
# print('重心：['+str(gx2)+','+str(gy2)+']' )
# stress_x2, stress_y2 = stress(img2, gx2, gy2)
# print('压力变化特征：[' + str(stress_x2) + ',' + str(stress_y2) + ']')
# slant_x2, slant_y2 = slant(img2, gx2, gy2)
# print('倾斜平衡特征：[' + str(slant_x2) + ',' + str(slant_y2) + ']')

# print('=================================================================')

# image2 = cv.imread('19997.jpg')
# img2 = to_cv2_ostu(image2)
# gx2, gy2 = G(img2)
# print('重心：['+str(gx2)+','+str(gy2)+']' )
# stress_x2, stress_y2 = stress(img2, gx2, gy2)
# print('压力变化特征：[' + str(stress_x2) + ',' + str(stress_y2) + ']')
# slant_x2, slant_y2 = slant(img2, gx2, gy2)
# print('倾斜平衡特征：[' + str(slant_x2) + ',' + str(slant_y2) + ']')
# """
