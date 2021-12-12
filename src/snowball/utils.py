import numpy as np


def coerce_array(array):
    """Function to attempt to coerce a list or array like data type into an ndarray.

    Args:
        array (list like): Datatype for attempting to be generated as an ndarray.

    Returns:
        [ndarray]: ndarray of the input array
    """

    if type(array) is not np.ndarray:
        try:
            array = np.array(array)
        except Exception as e:
            print(f"Unable to convert {array} into numpy array.")

    return array


def l2_norm(array1, array2):
    """Calculate the l2 norm of two arrays

    Args:
        array1 (ndarray): Values for first vector.  Note: will try to coerce into ndarray.
        array2 (ndarray): Values for second vector.  Note: will try to coerce into ndarray.

    Returns:
        float: l2 norm value of both arrays.
    """
    array1 = coerce_array(array1)
    array2 = coerce_array(array2)    
    l2 = np.linalg.norm(array1 - array2)
    return l2


def cosine_similarity(array1, array2):
    """Function to input two arrays and calculate the cosine similarity.

    Args:
        array1 (ndarray): Values for first vector.  Note: will try to coerce into ndarray.
        array2 (ndarray): Values for second vector.  Note: will try to coerce into ndarray.

    Returns:
        float: cosine similarity score for the arrays.
    """
    array1 = coerce_array(array1)
    array2 = coerce_array(array2)
    cos_sim = np.dot(array1, array2)/(np.linalg.norm(array1)*np.linalg.norm(array2))
    return cos_sim