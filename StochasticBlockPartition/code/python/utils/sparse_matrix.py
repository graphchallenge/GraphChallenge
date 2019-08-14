import abc
from typing import List, Union, Any, Iterable, Tuple

import numpy as np

from .edge_count_updates import EdgeCountUpdates


Values = List[Union[Any, List[Any]]]
Columns = List[Union[int, List[int]]]
Index = Union[int, slice, Iterable, Tuple]
Number = (int, np.int64, np.int32, np.int16, np.int)
Array = (List, Iterable)


class IndexResult(object):
    def __init__(self, values: Values, rows: Columns, columns: Columns, master_shape: Tuple[int, int]) -> None:
        self.values = values
        self.rows = rows
        self.columns = columns
        self.shape = master_shape
    # End of __init__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IndexResult):
            return False
        return (self.values, self.rows, self.columns, self.shape) == (other.values, other.rows, other.columns, self.shape)
    # End of __eq__()

    def strip(self) -> 'IndexResult':
        """Strips empty rows from a result with multiple rows."""
        values = list()
        rows = list()
        columns = list()
        if self.values and isinstance(self.values[0], List):
            for row in range(len(self.values)):
                if self.values[row]:
                    values.append(self.values[row])
                    rows.append(self.rows[row])
                    columns.append(self.columns[row])
            return IndexResult(values, rows, columns, self.shape)
        else:
            return self
    # End of strip()

    def to_matrix(self) -> 'DictMatrix':
        """Creates a DictMatrix out of the indexing result.
        """
        return DictMatrix(values=self)
    # End of to_matrix()

    def to_array(self) -> np.ndarray:
        """Converts the values held therein into an array.
        """
        return np.asarray(self.values)
    # End of to_array()

    def toarray(self) -> np.ndarray:
        """Traditional, sparse-vector-to-dense-vector-style array conversion.
        """
        if self.shape[0] == 1:
            array = np.zeros(self.shape)
            for value, column in zip(self.values, self.columns):
                array[0,column] = value
        elif self.shape[1] == 1:
            array = np.zeros(self.shape)
            for value, row in zip(self.values, self.rows):
                array[row,0] = value
        else:
            raise NotImplementedError("This function is only implemented for 1D arrays")
        return array
    # End of toarray()

    def delete(self, rows: List[int]) -> 'IndexResult':
        """Sets the given rows to be empty.
        """
        for row in rows:
            del self.rows[row]
            del self.columns[row]
            del self.values[row]
            # self.rows[row] = list()
            # self.columns[row] = list()
            # self.values[row] = list()
        return self
    # End of delete()

    @staticmethod
    def empty(master_shape: Tuple[int, int] = (0,0)) -> 'IndexResult':
        """Creates an empty IndexResult.
        """
        return IndexResult(list(), list(), list(), master_shape)
    # End of empty()
# End of IndexResult()


class SparseMatrix(abc.ABC):
    """An abstract base class for Sparse Matrix implementations.
    """

    @abc.abstractmethod
    def getrow(self, row: int) -> np.array:
        """Returns the values in a given row as a dense vector.

        TODO: Optimize this to return values and indexes instead

            Parameters
            ----------
            row : int
                the index of the row

            Returns
            -------
            row_values : np.array
                the values stored in the row
        """
        pass

    @abc.abstractmethod
    def getcol(self, col: int) -> np.array:
        """Returns the values in a given column as a dense vector.

        TODO: Optimize this to return values and indexes instead

            Parameters
            ----------
            col : int
                the index of the column

            Returns
            -------
            col_values : np.array
                the values stored in the column
        """
        pass

    @abc.abstractmethod
    def __getitem__(self, index: Index) -> IndexResult:
        """Returns the values from the selected indices as an IndexResult.

            Paramters
            ---------
            index : Index
                the indices requested
            
            result : IndexResult
                the values corresponding to the requested indices
        """
        pass

    @abc.abstractmethod
    def update_edge_counts(self, current_block: int, proposed_block: int, edge_count_updates: EdgeCountUpdates):
        """Updates the edge counts for the current and proposed blocks. The edge_count_updates will be updated in place.

            Parameters
            ---------
            current_block : int
                the current block
            proposed_block : int
                the proposed block
            edge_count_updates : EdgeCountUpdates
                the updates to apply
        """
        pass

    @abc.abstractmethod
    def nonzero(self) -> Tuple[List[int], List[int]]:
        """Returns the row and column indices of all nonzero elements in the matrix.

            Returns
            -------
            rows : List[int]
                The row indices of the non-zero elements
            columns : List[int]
                The column indices of the non-zero elements
        """
        pass

    @abc.abstractmethod
    def values(self) -> List[int]:
        """Returns the non-zero values of this matrix.

            Returns
            -------
            values : List[int]
                The non-zero values in this matrix
        """
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def copy(self) -> 'SparseMatrix':
        """Returns a copy of this matrix.

            Returns
            -------
            mat_copy : SparseMatrix
                a copy of this matrix
        """
        pass

    @abc.abstractmethod
    def sub(self, index: Index, values: Union[int, List[int], List[List[int]]]):
        """Subtracts a value to all the elements corresponding to the given index.

            Parameters
            ----------
            index : Index
                the index of the element(s) from which to subtract the value
            value : int
                the value of the elements to subtract
        """
        pass

    @abc.abstractmethod
    def add(self, index: Index, values: Union[int, List[int], List[List[int]]]):
        """Adds a value to all the elements corresponding to the given index.

            Parameters
            ----------
            index : Index
                the index of the element(s) to which to add the value
            value : int
                the value of the elements to add
        """
        pass
# End of SparseMatrix()
