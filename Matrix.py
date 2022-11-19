import numpy as np


class Matrix:
    def matrix_transposition(matrix):
        transposition_mtrx = np.zeros([len(matrix[0]), len(matrix)])
        for column_number in range(len(transposition_mtrx[0])):
            for row_number in range(len(transposition_mtrx)):
                transposition_mtrx[row_number][column_number] = matrix[column_number][row_number]
        return transposition_mtrx

    def matrix_multiplucation(first_matrix, second_matrix):
        if (len(first_matrix[0]) == len(second_matrix)):
            result_mtrx = np.zeros([len(first_matrix), len(second_matrix[0])])
            for row_number in range(len(result_mtrx)):
                for column_number in range(len(result_mtrx[0])):
                    sum_elem = 0
                    for new_column in range(len(second_matrix)):
                        sum_elem += first_matrix[row_number][new_column] * second_matrix[new_column][column_number]
                    result_mtrx[row_number][column_number] = sum_elem
            return result_mtrx
        else:
            raise TypeError("It is impossible to multiply matrices")

    def matrix_substraction(first_matrix, second_matrix):
        if len(first_matrix[0]) == len(second_matrix[0]) and len(first_matrix[1]) == len(second_matrix[1]):
            result_mtrx = np.zeros([len(first_matrix), len(first_matrix[0])])
            for row_number in range(len(result_mtrx)):
                for column_number in range(len(result_mtrx[0])):
                    result_mtrx[row_number][column_number] = first_matrix[row_number][column_number] - second_matrix[row_number][column_number]
            return result_mtrx
        else:
            raise TypeError("It is impossible to multiply matrices")

    def matrix_sum(first_matrix, second_matrix):
        if len(first_matrix[0]) == len(second_matrix[0]) and len(first_matrix[1]) == len(second_matrix[1]):
            result_mtrx = np.zeros([len(first_matrix), len(first_matrix[0])])
            for row_number in range(len(result_mtrx)):
                for column_number in range(len(result_mtrx[0])):
                    result_mtrx[row_number][column_number] = first_matrix[row_number][column_number] + second_matrix[row_number][column_number]
            return result_mtrx
        else:
            raise TypeError("It is impossible to multiply matrices")