from typing import Tuple, List, Dict, Any, Union, Iterable, NamedTuple
from collections import namedtuple
from sortedcontainers import SortedDict

import scipy.sparse as sparse
import numpy as np

# from edge_count_updates import EdgeCountUpdates
from .sparse_matrix import SparseMatrix, IndexResult, Number, Index, Values, Array, Columns
from .edge_count_updates import EdgeCountUpdates


class DictTransposeMatrix(SparseMatrix):
    """Stores a matrix as a list of dictionaries, where each dictionary represents the non-zero values in a row. Also
    stores the transpose of this matrix, for easy access to the matrix's columns.
    """

    def __init__(self, shape: Tuple = (0,0), lil: sparse.lil_matrix = sparse.lil_matrix((0,0))) -> None:
        """Creates a DictTransposeMatrix with the given shape from the given lil matrix.

            Parameters
            ---------
            values : sparse.lil_matrix
                The lil matrix containing the values to be stored in this matrix
        """
        if shape == (0,0) and lil.shape == (0,0):
            raise ValueError("Need to pass in either a valid shape or matrix")
        if shape != (0,0):
            self.nrows, self.ncols = shape
        elif lil.shape != (0,0):
            self.nrows, self.ncols = lil.shape
        self.shape = (self.nrows, self.ncols)
        self._matrix = list()  # type: List[SortedDict[int, Any]]
        self._matrix_T = list()  # type: List[SortedDict[int, Any]]

        if lil.shape != (0,0):
            for col in range(self.ncols):
                self._matrix_T.append(SortedDict())
            for row_index, row in enumerate(zip(lil.data, lil.rows)):
                rowdict = SortedDict()
                data, columns = row
                for index in range(len(data)):
                    rowdict[columns[index]] = data[index]
                    self._matrix_T[columns[index]][row_index] = data[index]
                self._matrix.append(rowdict)
        else:
            for _ in range(self.nrows):
                self._matrix.append(SortedDict())
            for _ in range(self.ncols):
                self._matrix_T.append(SortedDict())
    # End of __init__()

    def __getitem__(self, index: Index) -> IndexResult:
        if isinstance(index, tuple):
            return self._getitem_tuple(index)
        else:
            return self._getitem_rows(index)
    # End of __getitem__()

    def _getitem_rows(self, indexes: Union[int, slice, Iterable[int]]) -> IndexResult:
        """
        """
        if isinstance(indexes, Number):
            result = self._getitem_int(indexes)
        elif isinstance(indexes, slice):
            result = self._getitem_slice(indexes)
        elif isinstance(indexes, Array):
            result = self._getitem_iterable(indexes)
        return result
    # End of _getitem_rows()

    def _getitem_int(self, index: int) -> IndexResult:
        """Returns the values and column indexes of the non-zero items in the given row of the matrix.

            Parameters
            ----------
            index : int
                the index of the row

            Returns
            -------
            result : IndexResult
                the values and columns of the indexed row
        """
        row = self._matrix[index]
        return IndexResult(list(row.values()), [index] * len(row), list(row.keys()), (1, self.shape[1]))
    # End of _getitem_int()

    def _getitem_slice(self, index: slice) -> IndexResult:
        """Returns the values and column indexes of the non-zero items in the given slice of rows of the matrix.

            Parameters
            ----------
            index : slice
                the index slice of the rows

            Returns
            -------
            result : IndexResult
                the values and columns of the non-zero elements in the given rows
        """
        row_indices = np.arange(self.nrows)[index]
        result = self._matrix[index]
        values = list()
        rows = list()
        columns = list()
        for row_index, row in zip(row_indices, result):
            values.append(list(row.values()))
            rows.append([row_index] * len(row))
            columns.append(list(row.keys()))
        return IndexResult(values, rows, columns, self.shape)
    # End of _getitem_slice()

    def _getitem_iterable(self, index: Iterable[int]) -> IndexResult:
        """Returns the values and column indexes of the non-zero items in the given list of rows of the matrix.

            Parameters
            ----------
            index : Iterable[int]
                the index list of the rows

            Returns
            -------
            result : IndexResult
                the values and columns of the non-zero elements in the given rows
        """
        result = [self._matrix[i] for i in index]
        values = list()
        rows = list()
        columns = list()
        for row_index, row in zip(index, result):
            values.append(list(row.values()))
            rows.append([row_index] * len(row))
            columns.append(list(row.keys()))
        return IndexResult(values, rows, columns, self.shape)
    # End of _getitem_slice()

    def _getitem_tuple(self, index: Tuple) -> IndexResult:
        """Returns the values and column indexes of the non-zero items in the given rows and columns of
        the matrix.

            Parameters
            ----------
            index : Tuple
                the index list of the rows

            Returns
            -------
            result : IndexResult
                the values and columns of the non-zero elements in the given rows and columns
        """
        row_index = index[0]
        uresult = self._getitem_rows(row_index)
        col_index = index[1]
        if isinstance(row_index, Number):  # only one row is returned
            result = self._filtercolumns(uresult.values, uresult.rows, uresult.columns, col_index)
            result.shape = (1, result.shape[1])
        else:  # result has multiple rows
            result = IndexResult.empty(self.shape)
            for row in range(len(uresult.values)):
                temp_result = self._filtercolumns(uresult.values[row], uresult.rows[row], uresult.columns[row],
                                                  col_index)
                result.values.append(temp_result.values)
                result.rows.append(temp_result.rows)
                result.columns.append(temp_result.columns)
            result.shape = (uresult.shape[0], temp_result.shape[1])
        return result
    # End of _getitem_tuple()

    def _filtercolumns(self, values: Values, rows: Columns, columns: Columns, index: Index) -> IndexResult:
        """Filters the values based on the columns in which they appear.

            Paramters
            ---------
            values : Values
                the values to be filtered
            columns : Columns
                the columns in which they appear
            index : Index
                the column indexes to filter by

            Returns
            -------
            result : IndexResult
                the filtered values and the columns in which they appear
        """
        if isinstance(index, Number):
            result = self._filtercolumns_int(values, rows, columns, index)
        elif isinstance(index, slice):
            result = self._filtercolumns_slice(values, rows, columns, index)
        elif isinstance(index, Array):
            result = self._filtercolumns_iterable(values, rows, columns, index)
        return result
    # End of _filtercolumns()

    def _filtercolumns_int(self, values: Values, rows: Columns, columns: Columns, index: int) -> IndexResult:
        """Filters values by column index.

            Parameters
            ----------
            values : List[Any]
                the non-zero values in a row
            columns : List[int]
                the columns in which the values are held
            index : int
                the column index to include

            Returns
            -------
            result : IndexResult
                the values and columns of the filtered non-zero elements in the given row
        """
        colindex = np.argwhere(np.asarray(columns) == index)
        if colindex.size:
            return IndexResult([values[colindex[0][0]]], [rows[0]], [columns[colindex[0][0]]], (self.shape[0], 1))
        else:
            return IndexResult.empty((self.shape[0], 1))
    # End of _filtercolumns_int()

    def _filtercolumns_slice(self, values: List[Any], rows: List[int], columns: List[int], index: slice) -> IndexResult:
        """Filters values by column index slice.

            Parameters
            ----------
            values : List[Any]
                the non-zero values in a row
            columns : List[int]
                the columns in which the values are held
            index : slice
                the column index slice to include

            Returns
            -------
            result : IndexResult
                the values and columns of the filtered non-zero elements in the given row
        """
        start = index.start if index.start is not None else 0
        stop = index.stop if index.stop is not None else max(columns) + 1 if columns else -1
        step = index.step if index.step is not None else 1
        expanded_index = np.arange(start, stop, step)
        colindex = list()
        for i, column in enumerate(columns):
            if column in expanded_index:
                colindex.append(i)
        if colindex:
            return IndexResult(np.asarray(values)[colindex].tolist(), rows[:len(colindex)],
                               np.asarray(columns)[colindex].tolist(), self.shape)
        else:
            return IndexResult.empty(self.shape)
    # End of _filtercolumns_int()

    def _filtercolumns_iterable(self, values: List[Any], rows: List[int], columns: List[int],
        index: Iterable[int]) -> IndexResult:
        """Filters values by list of column indexes.

            Parameters
            ----------
            values : List[Any]
                the non-zero values in a row
            columns : List[int]
                the columns in which the values are held
            index : Iterable[int]
                the column index slice to include

            Returns
            -------
            result : IndexResult
                the values and columns of the filtered non-zero elements in the given row
        """
        colindex = list()
        for i, column in enumerate(columns):
            if column in index:
                colindex.append(i)
        if colindex:
            return IndexResult(np.asarray(values)[colindex].tolist(), rows[:len(colindex)],
                               np.asarray(columns)[colindex].tolist(), self.shape)
        else:
            return IndexResult.empty(self.shape)
    # End of _filtercolumns_int()

    def add(self, index: Index, values: Union[int, List[int], List[List[int]]]):
        """Adds a value to all the elements corresponding to the given index.

            Parameters
            ----------
            index : Index
                the index of the element(s) to which to add the value
            value : int
                the value of the elements to add
        """
        if isinstance(index, tuple):
            row_index, col_index = index
            if isinstance(row_index, Number) and isinstance(col_index, Array) and isinstance(values, Array):
                self._add_int_list_list(row_index, col_index, values)
            elif isinstance(row_index, Number) and isinstance(col_index, Array) and isinstance(values, Number):
                self._add_int_list_int(row_index, col_index, values)
            elif isinstance(row_index, Number) and isinstance(col_index, Number) and isinstance(values, Number):
                self._add_int_int_int(row_index, col_index, values)
            elif isinstance(row_index, Array) and isinstance(col_index, Number) and isinstance(values, Array):
                self._add_list_int_list(row_index, col_index, values)
            elif isinstance(row_index, Array) and isinstance(col_index, Number) and isinstance(values, Number):
                self._add_list_int_int(row_index, col_index, values)
    # End of add()

    def _add_int_list_list(self, row_index: int, col_index: List[int], values: List[int]):
        """Equivalent of: matrix[0, [1, 2, 3]] += [4, 5, 6]
        """
        row = self._matrix[row_index]
        if len(values) != len(col_index):
            raise ValueError("Values and Column Indexes are not the same length: {} != {}".format(
                             len(values), len(col_index)))
        for index, column_index in enumerate(col_index):
            if column_index in row:
                row[column_index] += values[index]
            else:
                row[column_index] = values[index]
            column = self._matrix_T[column_index]
            if row_index in column:
                column[row_index] += values[index]
            else:
                column[row_index] = values[index]
    # End of _add_int_list_list()

    def _add_int_list_int(self, row_index: int, col_index: List[int], value: int):
        """Equivalent of: matrix[0, [1, 2, 3]] += 2
        """
        row = self._matrix[row_index]
        for column_index in col_index:
            if column_index in row:
                row[column_index] += value
            else:
                row[column_index] = value
            column = self._matrix_T[column_index]
            if row_index in column:
                column[row_index] += value
            else:
                column[row_index] = value
    # End of _add_int_list_list()

    def _add_int_int_int(self, row_index: int, col_index: int, value: int):
        """Equivalent of: matrix[0, 2] += 3
        """
        row = self._matrix[row_index]
        if col_index in row:
            row[col_index] += value
        else:
            row[col_index] = value
        column = self._matrix_T[col_index]
        if row_index in column:
            column[row_index] += value
        else:
            column[row_index] = value
    # End of _add_int_list_list()

    def _add_list_int_list(self, row_index: List[int], col_index: int, values: List[int]):
        """Equivalent of: matrix[[1, 2, 3], 1] += [4, 5, 6]
        """
        if len(values) != len(row_index):
            raise ValueError("Values and Row Indexes are not the same length: {} != {}".format(
                             len(values), len(row_index)))
        column = self._matrix_T[col_index]
        for index, row in enumerate(row_index):
            if col_index in self._matrix[row]:
                self._matrix[row][col_index] += values[index]
            else:
                self._matrix[row][col_index] = values[index]
            if row in column:
                column[row] += values[index]
            else:
                column[row] = values[index]
    # End of _add_int_list_list()

    def _add_list_int_int(self, row_index: List[int], col_index: int, value: int):
        """Equivalent of: matrix[[1, 2, 3], 1] += 2
        """
        column = self._matrix_T[col_index]
        for row in row_index:
            if col_index in self._matrix[row]:
                self._matrix[row][col_index] += value
            else:
                self._matrix[row][col_index] = value
            if row in column:
                column[row] += value
            else:
                column[row] = value
    # End of _add_int_list_list()

    def sub(self, index: Index, values: Union[int, List[int], List[List[int]]]):
        """Subtracts a value to all the elements corresponding to the given index.

            Parameters
            ----------
            index : Index
                the index of the element(s) from which to subtract the value
            value : int
                the value of the elements to subtract
        """
        if isinstance(index, tuple):
            row_index, col_index = index
            if isinstance(row_index, Number) and isinstance(col_index, Array) and isinstance(values, Array):
                self._sub_int_list_list(row_index, col_index, values)
            elif isinstance(row_index, Number) and isinstance(col_index, Array) and isinstance(values, Number):
                self._sub_int_list_int(row_index, col_index, values)
            elif isinstance(row_index, Number) and isinstance(col_index, Number) and isinstance(values, Number):
                self._sub_int_int_int(row_index, col_index, values)
            elif isinstance(row_index, Array) and isinstance(col_index, Number) and isinstance(values, Array):
                self._sub_list_int_list(row_index, col_index, values)
            elif isinstance(row_index, Array) and isinstance(col_index, Number) and isinstance(values, Number):
                self._sub_list_int_int(row_index, col_index, values)
    # End of add()

    def _sub_int_list_list(self, row_index: int, col_index: List[int], values: List[int]):
        """Equivalent of: matrix[0, [1, 2, 3]] -= [4, 5, 6]
        """
        row = self._matrix[row_index]
        if len(values) != len(col_index):
            raise ValueError("Values and Column Indexes are not the same length: {} != {}".format(
                             len(values), len(col_index)))
        for index, column in enumerate(col_index):
            if column in row:
                row[column] -= values[index]
                if row[column] == 0:
                    row.pop(column)
            else:
                row[column] = -values[index]
                if row[column] == 0:
                    row.pop(column)
    # End of _add_int_list_list()

    def _sub_int_list_int(self, row_index: int, col_index: List[int], value: int):
        """Equivalent of: matrix[0, [1, 2, 3]] -= 2
        """
        row = self._matrix[row_index]
        for column in col_index:
            if column in row:
                row[column] -= value
                if row[column] == 0:
                    row.pop(column)
            else:
                row[column] = -value
                if row[column] == 0:
                    row.pop(column)
    # End of _add_int_list_list()

    def _sub_int_int_int(self, row_index: int, col_index: int, value: int):
        """Equivalent of: matrix[0, 2] -= 3
        """
        row = self._matrix[row_index]
        if col_index in row:
            row[col_index] -= value
            if row[col_index] == 0:
                row.pop(col_index)
        else:
            row[col_index] = -value
            if row[col_index] == 0:
                row.pop(col_index)
    # End of _add_int_list_list()

    def _sub_list_int_list(self, row_index: List[int], col_index: int, values: List[int]):
        """Equivalent of: matrix[[1, 2, 3], 1] -= [4, 5, 6]
        """
        if len(values) != len(row_index):
            raise ValueError("Values and Row Indexes are not the same length: {} != {}".format(
                             len(values), len(row_index)))
        for index, row in enumerate(row_index):
            if col_index in self._matrix[row]:
                self._matrix[row][col_index] -= values[index]
                if self._matrix[row][col_index] == 0:
                    self._matrix[row].pop(col_index)
            else:
                self._matrix[row][col_index] = -values[index]
                if self._matrix[row][col_index] == 0:
                    self._matrix[row].pop(col_index)
    # End of _add_int_list_list()

    def _sub_list_int_int(self, row_index: List[int], col_index: int, value: int):
        """Equivalent of: matrix[[1, 2, 3], 1] -= 2
        """
        for row in row_index:
            if col_index in self._matrix[row]:
                self._matrix[row][col_index] -= value
                if self._matrix[row][col_index] == 0:
                    self._matrix[row].pop(col_index)
            else:
                self._matrix[row][col_index] = -value
                if self._matrix[row][col_index] == 0:
                    self._matrix[row].pop(col_index)
    # End of _add_int_list_list()

    def sum(self, axis: Union[int, None] = None) -> Union[float, List[float]]:
        """Sums up all the values in the dictionary, or sums them up by rows or columns.

            Parameters
            ---------
            axis : Union[int, None]
                the axis along which to sum the matrix. Default = None. If None, sums all the values in the matrix.
                If 0, sums the values along each column. If 1, sums the values along each row.

            Returns
            -------
            result : Union[float, List[float]]
                the sum of the values in the matrix
        """
        if axis is None:
            result = 0.0
            for row in self._matrix:
                result += sum(row.values())
        elif axis == 1:  # sum across rows
            result = list()  # type: List[float]
            for row in self._matrix:
                result.append(sum(row.values()))
        elif axis == 0:  # sum across columns
            result = list()  # type: List[float]
            for col in self._matrix_T:
                result.append(sum(col.values()))
        return result
    # End of sum()

    def getrow(self, row: int):  # -> Tuple[List[int], List[int]]:
        result = np.zeros(self.ncols, dtype=int)
        rowdict = self._matrix[row]
        result[rowdict.keys()] = rowdict.values()
        return result
        # return list(rowdict.keys()), list(rowdict.values())
    # End of getrow()
    
    def getcol(self, col: int):  #  -> Tuple[List[int], List[int]]:
        result = np.zeros(self.nrows, dtype=int)
        coldict = self._matrix_T[col]
        result[coldict.keys()] = coldict.values()
        return result
    # End of getcol()

    def nonzero(self) -> Tuple[List[int], List[int]]:
        """Returns the row and column indices of all nonzero elements in the matrix.

            Returns
            -------
            rows : List[int]
                The row indices of the non-zero elements
            columns : List[int]
                The column indices of the non-zero elements
        """
        rows = list()  # type: List[int]
        columns = list()  # type: List[int]
        for row_index in range(self.nrows):
            row = self._matrix[row_index]
            rows.extend([row_index] * len(row))
            columns.extend(row.keys())
        return rows, columns
    # End of nonzero()

    def values(self) -> List[int]:
        """Returns the non-zero values of this matrix.

            Returns
            -------
            values : List[int]
                The non-zero values in this matrix
        """
        result = list()  # type: List[int]
        for row in self._matrix:
            result.extend(row.values())
        return result
    # End of values()

    def update_edge_counts(self, current_block: int, proposed_block: int, edge_count_updates: EdgeCountUpdates):
        """Updates the edge counts for the current and proposed blocks.

            Parameters
            ---------
            current_block : int
                the current block
            proposed_block : int
                the proposed block
            edge_count_updates : EdgeCountUpdates
                the updates to apply
        """
        # Deal with the normal matrix first
        self._matrix[current_block] = SortedDict()
        self._matrix[proposed_block] = SortedDict()
        for col in range(len(edge_count_updates.block_row)):
            r_value = edge_count_updates.block_row[col]
            s_value = edge_count_updates.proposal_row[col]
            if r_value != 0: self._matrix[current_block][col] = r_value
            if s_value != 0: self._matrix[proposed_block][col] = s_value
        for index in range(self.nrows):
            row = self._matrix[index]
            r_value = edge_count_updates.block_col[index]
            s_value = edge_count_updates.proposal_col[index]
            if r_value != 0:
                row[current_block] = r_value
            else:
                if current_block in row: row.pop(current_block)
            if s_value != 0:
                row[proposed_block] = s_value
            else:
                if proposed_block in row: row.pop(proposed_block)
        # Then deal with the transpose matrix
        # Columns and rows are switched (e.g.: block_col == block_row, and vice versa)
        self._matrix_T[current_block] = SortedDict()
        self._matrix_T[proposed_block] = SortedDict()
        for col in range(len(edge_count_updates.block_col)):
            r_value = edge_count_updates.block_col[col]
            s_value = edge_count_updates.proposal_col[col]
            if r_value != 0: self._matrix_T[current_block][col] = r_value
            if s_value != 0: self._matrix_T[proposed_block][col] = s_value
        for index in range(self.ncols):
            row = self._matrix_T[index]
            r_value = edge_count_updates.block_row[index]
            s_value = edge_count_updates.proposal_row[index]
            if r_value != 0:
                row[current_block] = r_value
            else:
                if current_block in row: row.pop(current_block)
            if s_value != 0:
                row[proposed_block] = s_value
            else:
                if proposed_block in row: row.pop(proposed_block)
    # End of update_edge_counts()

    def copy(self) -> 'DictTransposeMatrix':
        """Returns a copy of this matrix.

            Returns
            -------
            mat_copy : DictTransposeMatrix
                a copy of this matrix
        """
        mat_copy = DictTransposeMatrix(shape=self.shape)
        for index in range(mat_copy.nrows):
            mat_copy._matrix[index] = self._matrix[index].copy()
        for index in range(mat_copy.ncols):
            mat_copy._matrix_T[index] = self._matrix_T[index].copy()
        return mat_copy
    # End of copy()
# End of DictMatrix()

if __name__ == "__main__":
    # matrix               matrix_T
    # 0 1 0 0              0 1 0 0 1
    # 1 0 0 0              1 0 0 0 1
    # 0 0 0 1              0 0 0 0 1
    # 0 0 0 0              0 0 1 0 1
    # 1 1 1 1
    lil = sparse.lil_matrix([[0,1,0,0], [1,0,0,0], [0,0,0,1], [0,0,0,0], [1,1,1,1]])
    dm = DictTransposeMatrix(lil=lil)
    assert(dm._matrix == [{1: 1}, {0: 1}, {3: 1}, {}, {0: 1, 1: 1, 2: 1, 3: 1}])
    assert(dm._matrix_T == [{1: 1, 4: 1}, {0: 1, 4: 1}, {4: 1}, {2: 1, 4: 1}])

    #####################################
    # GETITEM tests
    #####################################
    res = dm[0]
    assert(res.values == [1] and res.columns == [1] and res.rows == [0])
    res = dm[3]
    assert(res.values == [] and res.columns == [] and res.rows == [])
    res = dm[4]
    assert(res.values == [1, 1, 1, 1] and res.columns == [0, 1, 2, 3] and res.rows == [4, 4, 4, 4])
    res = dm[0:2]
    assert(res.values == [[1], [1]] and res.columns == [[1], [0]] and res.rows == [[0], [1]])
    res = dm[:]
    assert(res.values == [[1], [1], [1], [], [1,1,1,1]] and res.columns == [[1], [0], [3], [], [0,1,2,3]] and res.rows == [[0], [1], [2], [], [4, 4, 4, 4]])
    res = dm[[0,1]]
    assert(res.values == [[1], [1]] and res.columns == [[1], [0]] and res.rows == [[0], [1]])
    res = dm[[0,1,2,3,4]]
    assert(res.values == [[1], [1], [1], [], [1,1,1,1]] and res.columns == [[1], [0], [3], [], [0,1,2,3]] and res.rows == [[0], [1], [2], [], [4, 4, 4, 4]])
    res = dm[0,0]
    assert(res.values == [] and res.columns == [] and res.rows == [])
    res = dm[0,1]
    assert(res.values == [1] and res.columns == [1] and res.rows == [0])
    res = dm[2,3]
    assert(res.values == [1] and res.columns == [3] and res.rows == [2])
    res = dm[0,0:2]
    assert(res.values == [1] and res.columns == [1] and res.rows == [0])
    res = dm[0,:]
    assert(res.values == [1] and res.columns == [1] and res.rows == [0])
    res = dm[3,:]
    assert(res.values == [] and res.columns == [] and res.rows == [])
    res = dm[3, 0:4]
    assert(res.values == [] and res.columns == [] and res.rows == [])
    res = dm[3, [0,1,2,3]]
    assert(res.values == [] and res.columns == [] and res.rows == [])
    res = dm[0,[0,1,2,3]]
    assert(res.values == [1] and res.columns == [1] and res.rows == [0])
    res = dm[[0,1,2,3,4],[0,1,2,3,4]]
    assert(res.values == [[1], [1], [1], [], [1,1,1,1]] and res.columns == [[1], [0], [3], [], [0,1,2,3]] and res.rows == [[0], [1], [2], [], [4, 4, 4, 4]])
    res = dm[:,:]
    assert(res.values == [[1], [1], [1], [], [1,1,1,1]] and res.columns == [[1], [0], [3], [], [0,1,2,3]] and res.rows == [[0], [1], [2], [], [4, 4, 4, 4]])
    ##############################
    # SUMMATION tests
    ##############################
    res = dm.sum()
    assert(res == 7.0)
    res = dm.sum(axis=0)
    assert(res == [2.0, 2.0, 1.0, 2.0])
    res = dm.sum(axis=1)
    assert(res == [1.0, 1.0, 1.0, 0.0, 4.0])
    ##############################
    # ADDITION/SUBTRACTION tests
    ##############################
    dm.add((0,[0,1,3]), [1,1,1])
    res = dm[0]
    assert(res.values == [1,2,1] and res.columns == [0,1,3])
    dm.add((0,[0,1,3]), 2)
    res = dm[0]
    assert(res.values == [3,4,3] and res.columns == [0,1,3])
    dm.add((0,2), 1)
    res = dm[0]
    assert(res.values == [3,4,1,3] and res.columns == [0,1,2,3])
    dm.add(([0,1,2], 1), [1, 1, 1])
    res = dm[[0,1,2],1]
    assert(res.values == [[5], [1], [1]] and res.columns == [[1], [1], [1]])
    dm.add(([1,2], 2), 1)
    res = dm[[1,2],2]
    assert(res.values == [[1], [1]] and res.columns == [[2], [2]])
    
    dm.sub((0,[0,1,3]), [1,1,1])
    res = dm[0]
    assert(res.values == [2,4,1,2] and res.columns == [0,1,2,3])
    dm.sub((0,[0,1,3]), 2)
    res = dm[0]
    assert(res.values == [2,1] and res.columns == [1,2])
    dm.sub((0,2), 1)
    res = dm[0]
    assert(res.values == [2] and res.columns == [1])
    dm.sub(([0,1,2], 1), [1, 1, 1])
    res = dm[[0,1,2],1]
    assert(res.values == [[1], [], []] and res.columns == [[1], [], []])
    dm.sub(([1,2], 2), 1)
    res = dm[[1,2],2]
    assert(res.values == [[], []] and res.columns == [[], []])
