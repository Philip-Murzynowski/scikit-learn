
#------------------------------------------------------------------------------

# Authors: Danny Sullivan <dbsullivan23@gmail.com>
#          Tom Dupre la Tour <tom.dupre-la-tour@m4x.org>
#          Arthur Mensch <arthur.mensch@m4x.org
#
# License: BSD 3 clause

"""
SAG and SAGA implementation
WARNING: Do not edit .pyx file directly, it is generated from .pyx.tp
"""

cimport numpy as cnp
import numpy as np
from libc.math cimport fabs, exp, log
from libc.time cimport time, time_t

from ._sgd_fast cimport LossFunction
from ._sgd_fast cimport Log, SquaredLoss

from ..metrics._ranking import roc_curve, auc

from ..utils._seq_dataset cimport SequentialDataset32, SequentialDataset64

from libc.stdio cimport printf

cnp.import_array()


cdef extern from "_sgd_fast_helpers.h":
    bint skl_isfinite64(double) nogil


cdef extern from "_sgd_fast_helpers.h":
    bint skl_isfinite32(float) nogil


cdef inline double fmax64(double x, double y) nogil:
    if x > y:
        return x
    return y

cdef inline float fmax32(float x, float y) nogil:
    if x > y:
        return x
    return y

cdef double _logsumexp64(double* arr, int n_classes) nogil:
    """Computes the sum of arr assuming arr is in the log domain.

    Returns log(sum(exp(arr))) while minimizing the possibility of
    over/underflow.
    """
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    cdef double vmax = arr[0]
    cdef double out = 0.0
    cdef int i

    for i in range(1, n_classes):
        if vmax < arr[i]:
            vmax = arr[i]

    for i in range(n_classes):
        out += exp(arr[i] - vmax)

    return log(out) + vmax

cdef float _logsumexp32(float* arr, int n_classes) nogil:
    """Computes the sum of arr assuming arr is in the log domain.

    Returns log(sum(exp(arr))) while minimizing the possibility of
    over/underflow.
    """
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    cdef float vmax = arr[0]
    cdef float out = 0.0
    cdef int i

    for i in range(1, n_classes):
        if vmax < arr[i]:
            vmax = arr[i]

    for i in range(n_classes):
        out += exp(arr[i] - vmax)

    return log(out) + vmax

cdef class MultinomialLogLoss64:
    cdef double _loss(self, double* prediction, double y, int n_classes,
                      double sample_weight) nogil:
        r"""Multinomial Logistic regression loss.

        The multinomial logistic loss for one sample is:
        loss = - sw \sum_c \delta_{y,c} (prediction[c] - logsumexp(prediction))
             = sw (logsumexp(prediction) - prediction[y])

        where:
            prediction = dot(x_sample, weights) + intercept
            \delta_{y,c} = 1 if (y == c) else 0
            sw = sample_weight

        Parameters
        ----------
        prediction : pointer to a np.ndarray[double] of shape (n_classes,)
            Prediction of the multinomial classifier, for current sample.

        y : double, between 0 and n_classes - 1
            Indice of the correct class for current sample (i.e. label encoded).

        n_classes : integer
            Total number of classes.

        sample_weight : double
            Weight of current sample.

        Returns
        -------
        loss : double
            Multinomial loss for current sample.

        Reference
        ---------
        Bishop, C. M. (2006). Pattern recognition and machine learning.
        Springer. (Chapter 4.3.4)
        """
        cdef double logsumexp_prediction = _logsumexp64(prediction, n_classes)
        cdef double loss

        # y is the indice of the correct class of current sample.
        loss = (logsumexp_prediction - prediction[int(y)]) * sample_weight
        return loss

    cdef void dloss(self, double* prediction, double y, int n_classes,
                     double sample_weight, double* gradient_ptr) nogil:
        r"""Multinomial Logistic regression gradient of the loss.

        The gradient of the multinomial logistic loss with respect to a class c,
        and for one sample is:
        grad_c = - sw * (p[c] - \delta_{y,c})

        where:
            p[c] = exp(logsumexp(prediction) - prediction[c])
            prediction = dot(sample, weights) + intercept
            \delta_{y,c} = 1 if (y == c) else 0
            sw = sample_weight

        Note that to obtain the true gradient, this value has to be multiplied
        by the sample vector x.

        Parameters
        ----------
        prediction : pointer to a np.ndarray[double] of shape (n_classes,)
            Prediction of the multinomial classifier, for current sample.

        y : double, between 0 and n_classes - 1
            Indice of the correct class for current sample (i.e. label encoded)

        n_classes : integer
            Total number of classes.

        sample_weight : double
            Weight of current sample.

        gradient_ptr : pointer to a np.ndarray[double] of shape (n_classes,)
            Gradient vector to be filled.

        Reference
        ---------
        Bishop, C. M. (2006). Pattern recognition and machine learning.
        Springer. (Chapter 4.3.4)
        """
        cdef double logsumexp_prediction = _logsumexp64(prediction, n_classes)
        cdef int class_ind

        for class_ind in range(n_classes):
            gradient_ptr[class_ind] = exp(prediction[class_ind] -
                                          logsumexp_prediction)

            # y is the indice of the correct class of current sample.
            if class_ind == y:
                gradient_ptr[class_ind] -= 1.0

            gradient_ptr[class_ind] *= sample_weight

    def __reduce__(self):
        return MultinomialLogLoss64, ()

cdef class MultinomialLogLoss32:
    cdef float _loss(self, float* prediction, float y, int n_classes,
                      float sample_weight) nogil:
        r"""Multinomial Logistic regression loss.

        The multinomial logistic loss for one sample is:
        loss = - sw \sum_c \delta_{y,c} (prediction[c] - logsumexp(prediction))
             = sw (logsumexp(prediction) - prediction[y])

        where:
            prediction = dot(x_sample, weights) + intercept
            \delta_{y,c} = 1 if (y == c) else 0
            sw = sample_weight

        Parameters
        ----------
        prediction : pointer to a np.ndarray[float] of shape (n_classes,)
            Prediction of the multinomial classifier, for current sample.

        y : float, between 0 and n_classes - 1
            Indice of the correct class for current sample (i.e. label encoded).

        n_classes : integer
            Total number of classes.

        sample_weight : float
            Weight of current sample.

        Returns
        -------
        loss : float
            Multinomial loss for current sample.

        Reference
        ---------
        Bishop, C. M. (2006). Pattern recognition and machine learning.
        Springer. (Chapter 4.3.4)
        """
        cdef float logsumexp_prediction = _logsumexp32(prediction, n_classes)
        cdef float loss

        # y is the indice of the correct class of current sample.
        loss = (logsumexp_prediction - prediction[int(y)]) * sample_weight
        return loss

    cdef void dloss(self, float* prediction, float y, int n_classes,
                     float sample_weight, float* gradient_ptr) nogil:
        r"""Multinomial Logistic regression gradient of the loss.

        The gradient of the multinomial logistic loss with respect to a class c,
        and for one sample is:
        grad_c = - sw * (p[c] - \delta_{y,c})

        where:
            p[c] = exp(logsumexp(prediction) - prediction[c])
            prediction = dot(sample, weights) + intercept
            \delta_{y,c} = 1 if (y == c) else 0
            sw = sample_weight

        Note that to obtain the true gradient, this value has to be multiplied
        by the sample vector x.

        Parameters
        ----------
        prediction : pointer to a np.ndarray[float] of shape (n_classes,)
            Prediction of the multinomial classifier, for current sample.

        y : float, between 0 and n_classes - 1
            Indice of the correct class for current sample (i.e. label encoded)

        n_classes : integer
            Total number of classes.

        sample_weight : float
            Weight of current sample.

        gradient_ptr : pointer to a np.ndarray[float] of shape (n_classes,)
            Gradient vector to be filled.

        Reference
        ---------
        Bishop, C. M. (2006). Pattern recognition and machine learning.
        Springer. (Chapter 4.3.4)
        """
        cdef float logsumexp_prediction = _logsumexp32(prediction, n_classes)
        cdef int class_ind

        for class_ind in range(n_classes):
            gradient_ptr[class_ind] = exp(prediction[class_ind] -
                                          logsumexp_prediction)

            # y is the indice of the correct class of current sample.
            if class_ind == y:
                gradient_ptr[class_ind] -= 1.0

            gradient_ptr[class_ind] *= sample_weight

    def __reduce__(self):
        return MultinomialLogLoss32, ()

cdef inline double _soft_thresholding64(double x, double shrinkage) nogil:
    return fmax64(x - shrinkage, 0) - fmax64(- x - shrinkage, 0)

cdef inline float _soft_thresholding32(float x, float shrinkage) nogil:
    return fmax32(x - shrinkage, 0) - fmax32(- x - shrinkage, 0)

def sag64(SequentialDataset64 dataset,
        cnp.ndarray[double, ndim=2, mode='c'] weights_array,
        cnp.ndarray[double, ndim=1, mode='c'] intercept_array,
        int n_samples,
        int n_features,
        int n_classes,
        double tol,
        int max_iter,
        str loss_function,
        double step_size,
        double alpha,
        double beta,
        cnp.ndarray[double, ndim=2, mode='c'] sum_gradient_init,
        cnp.ndarray[double, ndim=2, mode='c'] gradient_memory_init,
        cnp.ndarray[bint, ndim=1, mode='c'] seen_init,
        int num_seen,
        bint fit_intercept,
        cnp.ndarray[double, ndim=1, mode='c'] intercept_sum_gradient_init,
        double intercept_decay,
        bint saga,
        bint verbose):
    """Stochastic Average Gradient (SAG) and SAGA solvers.

    Used in Ridge and LogisticRegression.

    Reference
    ---------
    Schmidt, M., Roux, N. L., & Bach, F. (2013).
    Minimizing finite sums with the stochastic average gradient
    https://hal.inria.fr/hal-00860051/document
    (section 4.3)

    :arxiv:`Defazio, A., Bach F. & Lacoste-Julien S. (2014).
    "SAGA: A Fast Incremental Gradient Method With Support
    for Non-Strongly Convex Composite Objectives" <1407.0202>`
    """
    # the data pointer for x, the current sample
    cdef double *x_data_ptr = NULL
    # the index pointer for the column of the data
    cdef int *x_ind_ptr = NULL
    # the number of non-zero features for current sample
    cdef int xnnz = -1
    # the label value for current sample
    # the label value for current sample
    cdef double y
    # the sample weight
    cdef double sample_weight

    # helper variable for indexes
    cdef int f_idx, s_idx, feature_ind, class_ind, j
    # the number of pass through all samples
    cdef int n_iter = 0
    # helper to track iterations through samples
    cdef int sample_itr
    # the index (row number) of the current sample
    cdef int sample_ind

    # the maximum change in weights, used to compute stopping criteria
    cdef double max_change
    # a holder variable for the max weight, used to compute stopping criteria
    cdef double max_weight

    # the start time of the fit
    cdef time_t start_time
    # the end time of the fit
    cdef time_t end_time

    # precomputation since the step size does not change in this implementation
    cdef double wscale_update = 1.0 - step_size * alpha

    # vector of booleans indicating whether this sample has been seen
    cdef bint* seen = <bint*> seen_init.data

    # helper for cumulative sum
    cdef double cum_sum

    # the pointer to the coef_ or weights
    cdef double* weights = <double * >weights_array.data
    # the pointer to the intercept_array
    cdef double* intercept = <double * >intercept_array.data

    # the pointer to the intercept_sum_gradient
    cdef double* intercept_sum_gradient = \
        <double * >intercept_sum_gradient_init.data

    # the sum of gradients for each feature
    cdef double* sum_gradient = <double*> sum_gradient_init.data
    # the previously seen gradient for each sample
    cdef double* gradient_memory = <double*> gradient_memory_init.data

    # the cumulative sums needed for JIT params
    cdef cnp.ndarray[double, ndim=1] cumulative_sums_array = \
        np.empty(n_samples, dtype=np.float64, order="c")
    cdef double* cumulative_sums = <double*> cumulative_sums_array.data

    # the index for the last time this feature was updated
    cdef cnp.ndarray[int, ndim=1] feature_hist_array = \
        np.zeros(n_features, dtype=np.int32, order="c")
    cdef int* feature_hist = <int*> feature_hist_array.data

    # the previous weights to use to compute stopping criteria
    cdef cnp.ndarray[double, ndim=2] previous_weights_array = \
        np.zeros((n_features, n_classes), dtype=np.float64, order="c")
    cdef double* previous_weights = <double*> previous_weights_array.data

    cdef cnp.ndarray[double, ndim=1] prediction_array = \
        np.zeros(n_classes, dtype=np.float64, order="c")
    cdef double* prediction = <double*> prediction_array.data

    cdef cnp.ndarray[double, ndim=1] gradient_array = \
        np.zeros(n_classes, dtype=np.float64, order="c")
    cdef double* gradient = <double*> gradient_array.data

    # Intermediate variable that need declaration since cython cannot infer when templating
    cdef double val

    # Bias correction term in saga
    cdef double gradient_correction

    # the scalar used for multiplying z
    cdef double wscale = 1.0

    # return value (-1 if an error occurred, 0 otherwise)
    cdef int status = 0

    # the cumulative sums for each iteration for the sparse implementation
    cumulative_sums[0] = 0.0

    # the multipliative scale needed for JIT params
    cdef cnp.ndarray[double, ndim=1] cumulative_sums_prox_array
    cdef double* cumulative_sums_prox

    cdef bint prox = beta > 0 and saga

    # Loss function to optimize
    cdef LossFunction loss
    # Whether the loss function is multinomial
    cdef bint multinomial = False
    # Multinomial loss function
    cdef MultinomialLogLoss64 multiloss

    if loss_function == "multinomial":
        multinomial = True
        multiloss = MultinomialLogLoss64()
    elif loss_function == "log":
        loss = Log()
    elif loss_function == "squared":
        loss = SquaredLoss()
    else:
        raise ValueError("Invalid loss parameter: got %s instead of "
                         "one of ('log', 'squared', 'multinomial')"
                         % loss_function)

    if prox:
        cumulative_sums_prox_array = np.empty(n_samples,
                                              dtype=np.float64, order="c")
        cumulative_sums_prox = <double*> cumulative_sums_prox_array.data
    else:
        cumulative_sums_prox = NULL

    print('64')
    printf('n_classes: %d\n', n_classes)

    cdef cnp.ndarray[double, ndim=1] losses_array = \
        np.zeros(3, dtype=np.float64, order="c")
    cdef double* losses = <double*> losses_array.data
    cdef double predloss, l1loss, l2loss 

    with nogil:
        start_time = time(NULL)
        for n_iter in range(max_iter):
            if verbose:
                # DEBUG
                _binomial_loss_regularized_all_samples64(dataset, wscale, weights, intercept, n_samples, n_features, n_classes, alpha, beta, losses, prediction)
                predloss, l1loss, l2loss = losses[0], losses[1], losses[2]
                printf('<sklearn> predloss | l1loss | l2loss : %f, %f, %f\n', predloss, l1loss, l2loss)
            for sample_itr in range(n_samples):
                # extract a random sample
                sample_ind = dataset.random(&x_data_ptr, &x_ind_ptr, &xnnz,
                                              &y, &sample_weight)

                # cached index for gradient_memory
                s_idx = sample_ind * n_classes

                # update the number of samples seen and the seen array
                if seen[sample_ind] == 0:
                    num_seen += 1
                    seen[sample_ind] = 1

                # make the weight updates
                if sample_itr > 0:
                   status = lagged_update64(weights, wscale, xnnz,
                                                  n_samples, n_classes,
                                                  sample_itr,
                                                  cumulative_sums,
                                                  cumulative_sums_prox,
                                                  feature_hist,
                                                  prox,
                                                  sum_gradient,
                                                  x_ind_ptr,
                                                  False,
                                                  n_iter)
                   if status == -1:
                       break

                # find the current prediction
                predict_sample64(x_data_ptr, x_ind_ptr, xnnz, weights, wscale,
                                       intercept, prediction, n_classes)

                # compute the gradient for this sample, given the prediction
                if multinomial:
                    multiloss.dloss(prediction, y, n_classes, sample_weight,
                                     gradient)
                else:
                    gradient[0] = loss.dloss(prediction[0], y) * sample_weight

                # L2 regularization by simply rescaling the weights
                wscale *= wscale_update

                # make the updates to the sum of gradients
                for j in range(xnnz):
                    feature_ind = x_ind_ptr[j]
                    val = x_data_ptr[j]
                    f_idx = feature_ind * n_classes
                    for class_ind in range(n_classes):
                        gradient_correction = \
                            val * (gradient[class_ind] -
                                   gradient_memory[s_idx + class_ind])
                        if saga:
                            weights[f_idx + class_ind] -= \
                                (gradient_correction * step_size
                                 * (1 - 1. / num_seen) / wscale)
                        sum_gradient[f_idx + class_ind] += gradient_correction

                # fit the intercept
                if fit_intercept:
                    for class_ind in range(n_classes):
                        gradient_correction = (gradient[class_ind] -
                                               gradient_memory[s_idx + class_ind])
                        intercept_sum_gradient[class_ind] += gradient_correction
                        gradient_correction *= step_size * (1. - 1. / num_seen)
                        if saga:
                            intercept[class_ind] -= \
                                (step_size * intercept_sum_gradient[class_ind] /
                                 num_seen * intercept_decay) + gradient_correction
                        else:
                            intercept[class_ind] -= \
                                (step_size * intercept_sum_gradient[class_ind] /
                                 num_seen * intercept_decay)

                        # check to see that the intercept is not inf or NaN
                        if not skl_isfinite64(intercept[class_ind]):
                            status = -1
                            break
                    # Break from the n_samples outer loop if an error happened
                    # in the fit_intercept n_classes inner loop
                    if status == -1:
                        break

                # update the gradient memory for this sample
                for class_ind in range(n_classes):
                    gradient_memory[s_idx + class_ind] = gradient[class_ind]

                if sample_itr == 0:
                    cumulative_sums[0] = step_size / (wscale * num_seen)
                    if prox:
                        cumulative_sums_prox[0] = step_size * beta / wscale
                else:
                    cumulative_sums[sample_itr] = \
                        (cumulative_sums[sample_itr - 1] +
                         step_size / (wscale * num_seen))
                    if prox:
                        cumulative_sums_prox[sample_itr] = \
                        (cumulative_sums_prox[sample_itr - 1] +
                             step_size * beta / wscale)
                # If wscale gets too small, we need to reset the scale.
                if wscale < 1e-9:
                    if verbose:
                        with gil:
                            print("rescaling...")
                    status = scale_weights64(
                        weights, &wscale, n_features, n_samples, n_classes,
                        sample_itr, cumulative_sums,
                        cumulative_sums_prox,
                        feature_hist,
                        prox, sum_gradient, n_iter)
                    if status == -1:
                        break

            # Break from the n_iter outer loop if an error happened in the
            # n_samples inner loop
            if status == -1:
                break

            # we scale the weights every n_samples iterations and reset the
            # just-in-time update system for numerical stability.
            status = scale_weights64(weights, &wscale, n_features,
                                           n_samples,
                                           n_classes, n_samples - 1,
                                           cumulative_sums,
                                           cumulative_sums_prox,
                                           feature_hist,
                                           prox, sum_gradient, n_iter)

            if status == -1:
                break
            # check if the stopping criteria is reached
            max_change = 0.0
            max_weight = 0.0
            for idx in range(n_features * n_classes):
                max_weight = fmax64(max_weight, fabs(weights[idx]))
                max_change = fmax64(max_change,
                                  fabs(weights[idx] -
                                       previous_weights[idx]))
                previous_weights[idx] = weights[idx]
            if ((max_weight != 0 and max_change / max_weight <= tol)
                or max_weight == 0 and max_change == 0):
                if verbose:
                    end_time = time(NULL)
                    with gil:
                        print("convergence after %d epochs took %d seconds" %
                              (n_iter + 1, end_time - start_time))
                break
            elif verbose:
                printf('Epoch %d, change: %.8f\n', n_iter + 1,
                                                  max_change / max_weight)
    n_iter += 1
    # We do the error treatment here based on error code in status to avoid
    # re-acquiring the GIL within the cython code, which slows the computation
    # when the sag/saga solver is used concurrently in multiple Python threads.
    if status == -1:
        raise ValueError(("Floating-point under-/overflow occurred at epoch"
                          " #%d. Scaling input data with StandardScaler or"
                          " MinMaxScaler might help.") % n_iter)

    if verbose and n_iter >= max_iter:
        end_time = time(NULL)
        print(("max_iter reached after %d seconds") %
              (end_time - start_time))

    return num_seen, n_iter

def sag32(SequentialDataset32 dataset,
        cnp.ndarray[float, ndim=2, mode='c'] weights_array,
        cnp.ndarray[float, ndim=1, mode='c'] intercept_array,
        int n_samples,
        int n_features,
        int n_classes,
        double tol,
        int max_iter,
        str loss_function,
        double step_size,
        double alpha,
        double beta,
        cnp.ndarray[float, ndim=2, mode='c'] sum_gradient_init,
        cnp.ndarray[float, ndim=2, mode='c'] gradient_memory_init,
        cnp.ndarray[bint, ndim=1, mode='c'] seen_init,
        int num_seen,
        bint fit_intercept,
        cnp.ndarray[float, ndim=1, mode='c'] intercept_sum_gradient_init,
        double intercept_decay,
        bint saga,
        bint verbose,
        cnp.ndarray[float, ndim=2, mode='c'] x_train,
        cnp.ndarray[float, ndim=1, mode='c'] y_train,
        cnp.ndarray[float, ndim=2, mode='c'] x_test,
        cnp.ndarray[float, ndim=1, mode='c'] y_test,
        ):
    """Stochastic Average Gradient (SAG) and SAGA solvers.

    Used in Ridge and LogisticRegression.

    Reference
    ---------
    Schmidt, M., Roux, N. L., & Bach, F. (2013).
    Minimizing finite sums with the stochastic average gradient
    https://hal.inria.fr/hal-00860051/document
    (section 4.3)

    :arxiv:`Defazio, A., Bach F. & Lacoste-Julien S. (2014).
    "SAGA: A Fast Incremental Gradient Method With Support
    for Non-Strongly Convex Composite Objectives" <1407.0202>`
    """

    # print('y_train')
    # for i in range(n_samples):
    #     printf('%f, ', y_train[i])
    # print()

    # the data pointer for x, the current sample
    cdef float *x_data_ptr = NULL
    # the index pointer for the column of the data
    cdef int *x_ind_ptr = NULL
    # the number of non-zero features for current sample
    cdef int xnnz = -1
    # the label value for current sample
    cdef float y
    # the sample weight
    cdef float sample_weight

    # helper variable for indexes
    cdef int f_idx, s_idx, feature_ind, class_ind, j
    # the number of pass through all samples
    cdef int n_iter = 0
    # helper to track iterations through samples
    cdef int sample_itr
    # the index (row number) of the current sample
    cdef int sample_ind

    # the maximum change in weights, used to compute stopping criteria
    cdef float max_change
    # a holder variable for the max weight, used to compute stopping criteria
    cdef float max_weight

    # the start time of the fit
    cdef time_t start_time
    # the end time of the fit
    cdef time_t end_time

    # precomputation since the step size does not change in this implementation
    cdef float wscale_update = 1.0 - step_size * alpha

    # vector of booleans indicating whether this sample has been seen
    cdef bint* seen = <bint*> seen_init.data

    # helper for cumulative sum
    cdef float cum_sum

    # the pointer to the coef_ or weights
    cdef float* weights = <float * >weights_array.data
    # the pointer to the intercept_array
    cdef float* intercept = <float * >intercept_array.data

    # the pointer to the intercept_sum_gradient
    cdef float* intercept_sum_gradient = \
        <float * >intercept_sum_gradient_init.data

    # the sum of gradients for each feature
    cdef float* sum_gradient = <float*> sum_gradient_init.data
    # the previously seen gradient for each sample
    cdef float* gradient_memory = <float*> gradient_memory_init.data

    # the cumulative sums needed for JIT params
    cdef cnp.ndarray[float, ndim=1] cumulative_sums_array = \
        np.empty(n_samples, dtype=np.float32, order="c")
    cdef float* cumulative_sums = <float*> cumulative_sums_array.data

    # the index for the last time this feature was updated
    cdef cnp.ndarray[int, ndim=1] feature_hist_array = \
        np.zeros(n_features, dtype=np.int32, order="c")
    cdef int* feature_hist = <int*> feature_hist_array.data

    # the previous weights to use to compute stopping criteria
    cdef cnp.ndarray[float, ndim=2] previous_weights_array = \
        np.zeros((n_features, n_classes), dtype=np.float32, order="c")
    cdef float* previous_weights = <float*> previous_weights_array.data

    cdef cnp.ndarray[float, ndim=1] prediction_array = \
        np.zeros(n_classes, dtype=np.float32, order="c")
    cdef float* prediction = <float*> prediction_array.data

    cdef cnp.ndarray[float, ndim=1] gradient_array = \
        np.zeros(n_classes, dtype=np.float32, order="c")
    cdef float* gradient = <float*> gradient_array.data

    # Intermediate variable that need declaration since cython cannot infer when templating
    cdef float val

    # Bias correction term in saga
    cdef float gradient_correction

    # the scalar used for multiplying z
    cdef float wscale = 1.0

    # return value (-1 if an error occurred, 0 otherwise)
    cdef int status = 0

    # the cumulative sums for each iteration for the sparse implementation
    cumulative_sums[0] = 0.0

    # the multipliative scale needed for JIT params
    cdef cnp.ndarray[float, ndim=1] cumulative_sums_prox_array
    cdef float* cumulative_sums_prox

    cdef bint prox = beta > 0 and saga

    # Loss function to optimize
    cdef LossFunction loss
    # Whether the loss function is multinomial
    cdef bint multinomial = False
    # Multinomial loss function
    cdef MultinomialLogLoss32 multiloss

    if loss_function == "multinomial":
        multinomial = True
        multiloss = MultinomialLogLoss32()
    elif loss_function == "log":
        loss = Log()
    elif loss_function == "squared":
        loss = SquaredLoss()
    else:
        raise ValueError("Invalid loss parameter: got %s instead of "
                         "one of ('log', 'squared', 'multinomial')"
                         % loss_function)

    if prox:
        cumulative_sums_prox_array = np.empty(n_samples,
                                              dtype=np.float32, order="c")
        cumulative_sums_prox = <float*> cumulative_sums_prox_array.data
    else:
        cumulative_sums_prox = NULL

    print('32')
    printf('n_classes: %d\n', n_classes)

    cdef cnp.ndarray[float, ndim=1] losses_array = \
        np.zeros(3, dtype=np.float32, order="c")
    cdef float* losses = <float*> losses_array.data
    cdef float predloss, l1loss, l2loss 
    cdef cnp.ndarray[float, ndim=1] predictions_array = \
        np.zeros(n_samples, dtype=np.float32, order="c")

    with nogil:
        start_time = time(NULL)
        for n_iter in range(max_iter):
            if verbose:
                # DEBUG
                _binomial_loss_regularized_all_samples32(dataset, wscale, weights, intercept, n_samples, n_features, n_classes, alpha, beta, losses, prediction)
                predloss, l1loss, l2loss = losses[0], losses[1], losses[2]
                printf('<sklearn> predloss | l1loss | l2loss : %f, %f, %f\n', predloss, l1loss, l2loss)
                with gil:
                    train_auROC = _auROC_all_samples32(x_train, wscale, weights_array, intercept_array, n_samples, n_features, n_classes, predictions_array, y_train)
                    print(f'<sklearn> [train_auROC]: {train_auROC}')
                    test_auROC = _auROC_all_samples32(x_test, wscale, weights_array, intercept_array, n_samples, n_features, n_classes, predictions_array, y_test)
                    print(f'<sklearn> [test_auROC]: {test_auROC}')
            for sample_itr in range(n_samples):
                # extract a random sample
                sample_ind = dataset.random(&x_data_ptr, &x_ind_ptr, &xnnz,
                                              &y, &sample_weight)

                # cached index for gradient_memory
                s_idx = sample_ind * n_classes

                # update the number of samples seen and the seen array
                if seen[sample_ind] == 0:
                    num_seen += 1
                    seen[sample_ind] = 1

                # make the weight updates
                if sample_itr > 0:
                   status = lagged_update32(weights, wscale, xnnz,
                                                  n_samples, n_classes,
                                                  sample_itr,
                                                  cumulative_sums,
                                                  cumulative_sums_prox,
                                                  feature_hist,
                                                  prox,
                                                  sum_gradient,
                                                  x_ind_ptr,
                                                  False,
                                                  n_iter)
                   if status == -1:
                       break

                # find the current prediction
                predict_sample32(x_data_ptr, x_ind_ptr, xnnz, weights, wscale,
                                       intercept, prediction, n_classes)

                # compute the gradient for this sample, given the prediction
                if multinomial:
                    multiloss.dloss(prediction, y, n_classes, sample_weight,
                                     gradient)
                else:
                    gradient[0] = loss.dloss(prediction[0], y) * sample_weight

                # L2 regularization by simply rescaling the weights
                wscale *= wscale_update
                # printf("wscale: %f, %e\n", wscale, wscale)
                # printf("wscale_update: %f, %e\n", wscale_update, wscale_update)

                # make the updates to the sum of gradients
                for j in range(xnnz):
                    feature_ind = x_ind_ptr[j]
                    val = x_data_ptr[j]
                    f_idx = feature_ind * n_classes
                    for class_ind in range(n_classes):
                        gradient_correction = \
                            val * (gradient[class_ind] -
                                   gradient_memory[s_idx + class_ind])
                        if saga:
                            weights[f_idx + class_ind] -= \
                                (gradient_correction * step_size
                                 * (1 - 1. / num_seen) / wscale)
                        sum_gradient[f_idx + class_ind] += gradient_correction

                # fit the intercept
                if fit_intercept:
                    for class_ind in range(n_classes):
                        gradient_correction = (gradient[class_ind] -
                                               gradient_memory[s_idx + class_ind])
                        intercept_sum_gradient[class_ind] += gradient_correction
                        gradient_correction *= step_size * (1. - 1. / num_seen)
                        if saga:
                            intercept[class_ind] -= \
                                (step_size * intercept_sum_gradient[class_ind] /
                                 num_seen * intercept_decay) + gradient_correction
                        else:
                            intercept[class_ind] -= \
                                (step_size * intercept_sum_gradient[class_ind] /
                                 num_seen * intercept_decay)

                        # check to see that the intercept is not inf or NaN
                        if not skl_isfinite32(intercept[class_ind]):
                            status = -1
                            break
                    # Break from the n_samples outer loop if an error happened
                    # in the fit_intercept n_classes inner loop
                    if status == -1:
                        break

                # update the gradient memory for this sample
                for class_ind in range(n_classes):
                    gradient_memory[s_idx + class_ind] = gradient[class_ind]

                if sample_itr == 0:
                    cumulative_sums[0] = step_size / (wscale * num_seen)
                    if prox:
                        cumulative_sums_prox[0] = step_size * beta / wscale
                else:
                    cumulative_sums[sample_itr] = \
                        (cumulative_sums[sample_itr - 1] +
                         step_size / (wscale * num_seen))
                    if prox:
                        cumulative_sums_prox[sample_itr] = \
                        (cumulative_sums_prox[sample_itr - 1] +
                             step_size * beta / wscale)
                # If wscale gets too small, we need to reset the scale.
                if wscale < 1e-9:
                    if verbose:
                        with gil:
                            print("rescaling...")
                    status = scale_weights32(
                        weights, &wscale, n_features, n_samples, n_classes,
                        sample_itr, cumulative_sums,
                        cumulative_sums_prox,
                        feature_hist,
                        prox, sum_gradient, n_iter)
                    if status == -1:
                        break

            # Break from the n_iter outer loop if an error happened in the
            # n_samples inner loop
            if status == -1:
                break

            # we scale the weights every n_samples iterations and reset the
            # just-in-time update system for numerical stability.
            status = scale_weights32(weights, &wscale, n_features,
                                           n_samples,
                                           n_classes, n_samples - 1,
                                           cumulative_sums,
                                           cumulative_sums_prox,
                                           feature_hist,
                                           prox, sum_gradient, n_iter)

            if status == -1:
                break
            # check if the stopping criteria is reached
            max_change = 0.0
            max_weight = 0.0
            for idx in range(n_features * n_classes):
                max_weight = fmax32(max_weight, fabs(weights[idx]))
                max_change = fmax32(max_change,
                                  fabs(weights[idx] -
                                       previous_weights[idx]))
                previous_weights[idx] = weights[idx]
            if ((max_weight != 0 and max_change / max_weight <= tol)
                or max_weight == 0 and max_change == 0):
                if verbose:
                    end_time = time(NULL)
                    with gil:
                        print("convergence after %d epochs took %d seconds" %
                              (n_iter + 1, end_time - start_time))
                break
            elif verbose:
                printf('Epoch %d, change: %.8f\n', n_iter + 1,
                                                  max_change / max_weight)
    if verbose:
        # DEBUG
        print('final')
        _binomial_loss_regularized_all_samples32(dataset, wscale, weights, intercept, n_samples, n_features, n_classes, alpha, beta, losses, prediction)
        predloss, l1loss, l2loss = losses[0], losses[1], losses[2]
        printf('<sklearn> [loss]: predloss | l1loss | l2loss : %f, %f, %f\n', predloss, l1loss, l2loss)
        train_auROC = _auROC_all_samples32(x_train, wscale, weights_array, intercept_array, n_samples, n_features, n_classes, predictions_array, y_train)
        print(f'<sklearn> [train_auROC]: {train_auROC}')
        test_auROC = _auROC_all_samples32(x_test, wscale, weights_array, intercept_array, n_samples, n_features, n_classes, predictions_array, y_test)
        print(f'<sklearn> [test_auROC]: {test_auROC}')
    n_iter += 1
    # We do the error treatment here based on error code in status to avoid
    # re-acquiring the GIL within the cython code, which slows the computation
    # when the sag/saga solver is used concurrently in multiple Python threads.
    if status == -1:
        raise ValueError(("Floating-point under-/overflow occurred at epoch"
                          " #%d. Scaling input data with StandardScaler or"
                          " MinMaxScaler might help.") % n_iter)

    if verbose and n_iter >= max_iter:
        end_time = time(NULL)
        print(("max_iter reached after %d seconds") %
              (end_time - start_time))

    return num_seen, n_iter

cdef int scale_weights64(double* weights, double* wscale,
                               int n_features,
                               int n_samples, int n_classes, int sample_itr,
                               double* cumulative_sums,
                               double* cumulative_sums_prox,
                               int* feature_hist,
                               bint prox,
                               double* sum_gradient,
                               int n_iter) nogil:
    """Scale the weights with wscale for numerical stability.

    wscale = (1 - step_size * alpha) ** (n_iter * n_samples + sample_itr)
    can become very small, so we reset it every n_samples iterations to 1.0 for
    numerical stability. To be able to scale, we first need to update every
    coefficients and reset the just-in-time update system.
    This also limits the size of `cumulative_sums`.
    """

    cdef int status
    status = lagged_update64(weights, wscale[0], n_features,
                                   n_samples, n_classes, sample_itr + 1,
                                   cumulative_sums,
                                   cumulative_sums_prox,
                                   feature_hist,
                                   prox,
                                   sum_gradient,
                                   NULL,
                                   True,
                                   n_iter)
    # if lagged update succeeded, reset wscale to 1.0
    if status == 0:
        wscale[0] = 1.0
    return status

cdef int scale_weights32(float* weights, float* wscale,
                               int n_features,
                               int n_samples, int n_classes, int sample_itr,
                               float* cumulative_sums,
                               float* cumulative_sums_prox,
                               int* feature_hist,
                               bint prox,
                               float* sum_gradient,
                               int n_iter) nogil:
    """Scale the weights with wscale for numerical stability.

    wscale = (1 - step_size * alpha) ** (n_iter * n_samples + sample_itr)
    can become very small, so we reset it every n_samples iterations to 1.0 for
    numerical stability. To be able to scale, we first need to update every
    coefficients and reset the just-in-time update system.
    This also limits the size of `cumulative_sums`.
    """

    cdef int status
    status = lagged_update32(weights, wscale[0], n_features,
                                   n_samples, n_classes, sample_itr + 1,
                                   cumulative_sums,
                                   cumulative_sums_prox,
                                   feature_hist,
                                   prox,
                                   sum_gradient,
                                   NULL,
                                   True,
                                   n_iter)
    # if lagged update succeeded, reset wscale to 1.0
    if status == 0:
        wscale[0] = 1.0
    return status

cdef int lagged_update64(double* weights, double wscale, int xnnz,
                               int n_samples, int n_classes, int sample_itr,
                               double* cumulative_sums,
                               double* cumulative_sums_prox,
                               int* feature_hist,
                               bint prox,
                               double* sum_gradient,
                               int* x_ind_ptr,
                               bint reset,
                               int n_iter) nogil:
    """Hard perform the JIT updates for non-zero features of present sample.
    The updates that awaits are kept in memory using cumulative_sums,
    cumulative_sums_prox, wscale and feature_hist. See original SAGA paper
    (Defazio et al. 2014) for details. If reset=True, we also reset wscale to
    1 (this is done at the end of each epoch).
    """
    cdef int feature_ind, class_ind, idx, f_idx, lagged_ind, last_update_ind
    cdef double cum_sum, grad_step, prox_step, cum_sum_prox
    for feature_ind in range(xnnz):
        if not reset:
            feature_ind = x_ind_ptr[feature_ind]
        f_idx = feature_ind * n_classes

        cum_sum = cumulative_sums[sample_itr - 1]
        if prox:
            cum_sum_prox = cumulative_sums_prox[sample_itr - 1]
        if feature_hist[feature_ind] != 0:
            cum_sum -= cumulative_sums[feature_hist[feature_ind] - 1]
            if prox:
                cum_sum_prox -= cumulative_sums_prox[feature_hist[feature_ind] - 1]
        if not prox:
            for class_ind in range(n_classes):
                idx = f_idx + class_ind
                weights[idx] -= cum_sum * sum_gradient[idx]
                if reset:
                    weights[idx] *= wscale
                    if not skl_isfinite64(weights[idx]):
                        # returning here does not require the gil as the return
                        # type is a C integer
                        return -1
        else:
            for class_ind in range(n_classes):
                idx = f_idx + class_ind
                if fabs(sum_gradient[idx] * cum_sum) < cum_sum_prox:
                    # In this case, we can perform all the gradient steps and
                    # all the proximal steps in this order, which is more
                    # efficient than unrolling all the lagged updates.
                    # Idea taken from scikit-learn-contrib/lightning.
                    weights[idx] -= cum_sum * sum_gradient[idx]
                    weights[idx] = _soft_thresholding64(weights[idx],
                                                      cum_sum_prox)
                else:
                    last_update_ind = feature_hist[feature_ind]
                    if last_update_ind == -1:
                        last_update_ind = sample_itr - 1
                    for lagged_ind in range(sample_itr - 1,
                                   last_update_ind - 1, -1):
                        if lagged_ind > 0:
                            grad_step = (cumulative_sums[lagged_ind]
                               - cumulative_sums[lagged_ind - 1])
                            prox_step = (cumulative_sums_prox[lagged_ind]
                               - cumulative_sums_prox[lagged_ind - 1])
                        else:
                            grad_step = cumulative_sums[lagged_ind]
                            prox_step = cumulative_sums_prox[lagged_ind]
                        weights[idx] -= sum_gradient[idx] * grad_step
                        weights[idx] = _soft_thresholding64(weights[idx],
                                                          prox_step)

                if reset:
                    weights[idx] *= wscale
                    # check to see that the weight is not inf or NaN
                    if not skl_isfinite64(weights[idx]):
                        return -1
        if reset:
            feature_hist[feature_ind] = sample_itr % n_samples
        else:
            feature_hist[feature_ind] = sample_itr

    if reset:
        cumulative_sums[sample_itr - 1] = 0.0
        if prox:
            cumulative_sums_prox[sample_itr - 1] = 0.0

    return 0

cdef int lagged_update32(float* weights, float wscale, int xnnz,
                               int n_samples, int n_classes, int sample_itr,
                               float* cumulative_sums,
                               float* cumulative_sums_prox,
                               int* feature_hist,
                               bint prox,
                               float* sum_gradient,
                               int* x_ind_ptr,
                               bint reset,
                               int n_iter) nogil:
    """Hard perform the JIT updates for non-zero features of present sample.
    The updates that awaits are kept in memory using cumulative_sums,
    cumulative_sums_prox, wscale and feature_hist. See original SAGA paper
    (Defazio et al. 2014) for details. If reset=True, we also reset wscale to
    1 (this is done at the end of each epoch).
    """
    cdef int feature_ind, class_ind, idx, f_idx, lagged_ind, last_update_ind
    cdef float cum_sum, grad_step, prox_step, cum_sum_prox
    for feature_ind in range(xnnz):
        if not reset:
            feature_ind = x_ind_ptr[feature_ind]
        f_idx = feature_ind * n_classes

        cum_sum = cumulative_sums[sample_itr - 1]
        if prox:
            cum_sum_prox = cumulative_sums_prox[sample_itr - 1]
        if feature_hist[feature_ind] != 0:
            cum_sum -= cumulative_sums[feature_hist[feature_ind] - 1]
            if prox:
                cum_sum_prox -= cumulative_sums_prox[feature_hist[feature_ind] - 1]
        if not prox:
            for class_ind in range(n_classes):
                idx = f_idx + class_ind
                weights[idx] -= cum_sum * sum_gradient[idx]
                if reset:
                    weights[idx] *= wscale
                    if not skl_isfinite32(weights[idx]):
                        # returning here does not require the gil as the return
                        # type is a C integer
                        return -1
        else:
            for class_ind in range(n_classes):
                idx = f_idx + class_ind
                if fabs(sum_gradient[idx] * cum_sum) < cum_sum_prox:
                    # In this case, we can perform all the gradient steps and
                    # all the proximal steps in this order, which is more
                    # efficient than unrolling all the lagged updates.
                    # Idea taken from scikit-learn-contrib/lightning.
                    weights[idx] -= cum_sum * sum_gradient[idx]
                    weights[idx] = _soft_thresholding32(weights[idx],
                                                      cum_sum_prox)
                else:
                    last_update_ind = feature_hist[feature_ind]
                    if last_update_ind == -1:
                        last_update_ind = sample_itr - 1
                    for lagged_ind in range(sample_itr - 1,
                                   last_update_ind - 1, -1):
                        if lagged_ind > 0:
                            grad_step = (cumulative_sums[lagged_ind]
                               - cumulative_sums[lagged_ind - 1])
                            prox_step = (cumulative_sums_prox[lagged_ind]
                               - cumulative_sums_prox[lagged_ind - 1])
                        else:
                            grad_step = cumulative_sums[lagged_ind]
                            prox_step = cumulative_sums_prox[lagged_ind]
                        weights[idx] -= sum_gradient[idx] * grad_step
                        weights[idx] = _soft_thresholding32(weights[idx],
                                                          prox_step)

                if reset:
                    weights[idx] *= wscale
                    # check to see that the weight is not inf or NaN
                    if not skl_isfinite32(weights[idx]):
                        return -1
        if reset:
            feature_hist[feature_ind] = sample_itr % n_samples
        else:
            feature_hist[feature_ind] = sample_itr

    if reset:
        cumulative_sums[sample_itr - 1] = 0.0
        if prox:
            cumulative_sums_prox[sample_itr - 1] = 0.0

    return 0

cdef void predict_sample64(double* x_data_ptr, int* x_ind_ptr, int xnnz,
                                 double* w_data_ptr, double wscale,
                                 double* intercept, double* prediction,
                                 int n_classes) nogil:
    """Compute the prediction given sparse sample x and dense weight w.

    Parameters
    ----------
    x_data_ptr : pointer
        Pointer to the data of the sample x

    x_ind_ptr : pointer
        Pointer to the indices of the sample  x

    xnnz : int
        Number of non-zero element in the sample  x

    w_data_ptr : pointer
        Pointer to the data of the weights w

    wscale : double
        Scale of the weights w

    intercept : pointer
        Pointer to the intercept

    prediction : pointer
        Pointer to store the resulting prediction

    n_classes : int
        Number of classes in multinomial case. Equals 1 in binary case.

    """
    cdef int feature_ind, class_ind, j
    cdef double innerprod

    for class_ind in range(n_classes):
        innerprod = 0.0
        # Compute the dot product only on non-zero elements of x
        for j in range(xnnz):
            feature_ind = x_ind_ptr[j]
            innerprod += (w_data_ptr[feature_ind * n_classes + class_ind] *
                          x_data_ptr[j])

        prediction[class_ind] = wscale * innerprod + intercept[class_ind]


cdef void predict_sample32(float* x_data_ptr, int* x_ind_ptr, int xnnz,
                                 float* w_data_ptr, float wscale,
                                 float* intercept, float* prediction,
                                 int n_classes) nogil:
    """Compute the prediction given sparse sample x and dense weight w.

    Parameters
    ----------
    x_data_ptr : pointer
        Pointer to the data of the sample x

    x_ind_ptr : pointer
        Pointer to the indices of the sample  x

    xnnz : int
        Number of non-zero element in the sample  x

    w_data_ptr : pointer
        Pointer to the data of the weights w

    wscale : float
        Scale of the weights w

    intercept : pointer
        Pointer to the intercept

    prediction : pointer
        Pointer to store the resulting prediction

    n_classes : int
        Number of classes in multinomial case. Equals 1 in binary case.

    """
    cdef int feature_ind, class_ind, j
    cdef float innerprod

    for class_ind in range(n_classes):
        innerprod = 0.0
        # Compute the dot product only on non-zero elements of x
        for j in range(xnnz):
            feature_ind = x_ind_ptr[j]
            innerprod += (w_data_ptr[feature_ind * n_classes + class_ind] *
                          x_data_ptr[j])

        prediction[class_ind] = wscale * innerprod + intercept[class_ind]

def _auROC_all_samples32(
        # SequentialDataset32 dataset, float wscale, dataset shuffled so do not use
        cnp.ndarray[float, ndim=2, mode='c'] x_train,
        float wscale,
        cnp.ndarray[float, ndim=2, mode='c'] weights_array,
        cnp.ndarray[float, ndim=1, mode='c'] intercept_array,
        int n_samples, int n_features, int n_classes,
        cnp.ndarray[float, ndim=1, mode='c'] predictions_array,
        cnp.ndarray[float, ndim=1, mode='c'] y_train):
    """ Compute auROC for all samples.

    Used for testing purpose only.
    predictions is an array which is passed in as a location for all prediction samples to be stored.
    """
    # the data pointer for x, the current sample
    cdef float *x_data_ptr = NULL
    # the index pointer for the column of the data
    cdef int *x_ind_ptr = NULL
    # the number of non-zero features for current sample
    cdef int xnnz = n_features
    # the label value for current sample
    cdef float y
    # the sample weight
    cdef float sample_weight
    # the pointer to the coef_ or weights
    cdef float* weights = <float * >weights_array.data
    # the pointer to the intercept_array
    cdef float* intercept = <float * >intercept_array.data
    # the pointer to the predictions_array
    # cdef float* predictions = <float * > predictions_array.data
    # the pointer the desired index in the predictions array
    cdef float* predictions_ptr
    # iterator
    cdef int i

    cdef cnp.ndarray[int, ndim=1, mode='c'] feature_indices = \
        np.arange(0, n_features, dtype=np.intc)
    x_ind_ptr = <int * >feature_indices.data

    # testing
    # printf('%p, %p | %p\n', <float*>(predictions_array.data), <float*>(predictions_array.data + 1), <float*>(<float*>(predictions_array.data) + 1))

    for i in range(n_samples):
        # # get next sample on the dataset
        # # y must be in {0, 1}.
        # dataset.next(&x_data_ptr, &x_ind_ptr, &xnnz,
        #              &y, &sample_weight)
        # #printf('y: %f\n', y)
        # #print(f'y_train[i]: {y_train[i]}')
        # predictions_ptr = <float * >(<float * >(predictions_array.data) + i)
        # # prediction of the multinomial classifier for the sample
        # predict_sample32(x_data_ptr, x_ind_ptr, xnnz, weights, wscale,
        #                intercept, predictions_ptr, n_classes)
        # get next sample on the dataset
        # y must be in {0, 1}.
        x_data_ptr = <float * >(<float * >(x_train.data) + i*n_features)
        predictions_ptr = <float * >(<float * >(predictions_array.data) + i)
        # prediction of the multinomial classifier for the sample
        predict_sample32(x_data_ptr, x_ind_ptr, xnnz, weights, wscale,
                       intercept, predictions_ptr, n_classes)
    predictions_array = 1.0 / (1.0 + np.exp(-1 * predictions_array))
    # convert y from {-1, 1} to {0, 1} as predictions will be in [0, 1]. necessary?
    fpr, tpr, _ = roc_curve(np.maximum(0, y_train), predictions_array)
    auROC = auc(fpr, tpr)
    return  auROC



def _multinomial_grad_loss_all_samples(
        SequentialDataset64 dataset,
        cnp.ndarray[double, ndim=2, mode='c'] weights_array,
        cnp.ndarray[double, ndim=1, mode='c'] intercept_array,
        int n_samples, int n_features, int n_classes):
    """Compute multinomial gradient and loss across all samples.

    Used for testing purpose only.
    """
    cdef double* weights = <double * >weights_array.data
    cdef double* intercept = <double * >intercept_array.data

    cdef double *x_data_ptr = NULL
    cdef int *x_ind_ptr = NULL
    cdef int xnnz = -1
    cdef double y
    cdef double sample_weight

    cdef double wscale = 1.0
    cdef int i, j, class_ind, feature_ind
    cdef double val
    cdef double sum_loss = 0.0

    cdef MultinomialLogLoss64 multiloss = MultinomialLogLoss64()

    cdef cnp.ndarray[double, ndim=2] sum_gradient_array = \
        np.zeros((n_features, n_classes), dtype=np.double, order="c")
    cdef double* sum_gradient = <double*> sum_gradient_array.data

    cdef cnp.ndarray[double, ndim=1] prediction_array = \
        np.zeros(n_classes, dtype=np.double, order="c")
    cdef double* prediction = <double*> prediction_array.data

    cdef cnp.ndarray[double, ndim=1] gradient_array = \
        np.zeros(n_classes, dtype=np.double, order="c")
    cdef double* gradient = <double*> gradient_array.data

    with nogil:
        for i in range(n_samples):
            # get next sample on the dataset
            dataset.next(&x_data_ptr, &x_ind_ptr, &xnnz,
                         &y, &sample_weight)

            # prediction of the multinomial classifier for the sample
            predict_sample64(x_data_ptr, x_ind_ptr, xnnz, weights, wscale,
                           intercept, prediction, n_classes)

            # compute the gradient for this sample, given the prediction
            multiloss.dloss(prediction, y, n_classes, sample_weight, gradient)

            # compute the loss for this sample, given the prediction
            sum_loss += multiloss._loss(prediction, y, n_classes, sample_weight)

            # update the sum of the gradient
            for j in range(xnnz):
                feature_ind = x_ind_ptr[j]
                val = x_data_ptr[j]
                for class_ind in range(n_classes):
                    sum_gradient[feature_ind * n_classes + class_ind] += \
                        gradient[class_ind] * val

    return sum_loss, sum_gradient_array

cdef double _binomial_loss_l1_64(
    int* x_ind_ptr, int xnnz,
    double* w_data_ptr, double wscale,
    double beta, int n_samples) nogil:
    """ Calculate l1 loss assuming the weights array only has 1 class.

    Note: Multiply by n_samples as beta has previously been divided by n_samples
    (actually using beta_scaled).
    """

    cdef double weights_sum
    cdef int feature_ind, j
    weights_sum = 0.0
    for j in range(xnnz):
        feature_ind = x_ind_ptr[j]
        weights_sum += fabs(w_data_ptr[feature_ind])
    return (n_samples * beta) * wscale * weights_sum 

cdef double _binomial_loss_l2_64(
    int* x_ind_ptr, int xnnz,
    double* w_data_ptr, double wscale,
    double alpha, int n_samples) nogil:
    """ Calculate l2 loss assuming the weights array only has 1 class.

    Note: Multiply by n_samples as alpha has previously been divided by n_samples.
    (actually using alpha_scaled).
    """

    cdef double weights_squared_sum, w
    cdef int feature_ind, j
    weights_sum = 0.0
    for j in range(xnnz):
        feature_ind = x_ind_ptr[j]
        w = w_data_ptr[feature_ind]
        weights_squared_sum += w * w
    return 0.5 * (n_samples * alpha) * (wscale * wscale) * weights_squared_sum 

cdef float _binomial_loss_l1_32(
    int* x_ind_ptr, int xnnz,
    float* w_data_ptr, float wscale,
    double beta, int n_samples) nogil:
    """ Calculate l1 loss assuming the weights array only has 1 class.

    Note: Multiply by n_samples as beta has previously been divided by n_samples
    (actually using beta_scaled).
    """

    cdef double weights_sum
    cdef int feature_ind, j
    weights_sum = 0.0
    for j in range(xnnz):
        feature_ind = x_ind_ptr[j]
        weights_sum += fabs(w_data_ptr[feature_ind])
    return (n_samples * beta) * wscale * weights_sum 

cdef float _binomial_loss_l2_32(
    int* x_ind_ptr, int xnnz,
    float* w_data_ptr, float wscale,
    double alpha, int n_samples) nogil:
    """ Calculate l2 loss assuming the weights array only has 1 class.

    Note: Multiply by n_samples as alpha has previously been divided by n_samples.
    (actually using alpha_scaled).
    """

    cdef double weights_squared_sum
    cdef float w
    cdef int feature_ind, j
    weights_squared_um = 0.0
    for j in range(xnnz):
        feature_ind = x_ind_ptr[j]
        w = w_data_ptr[feature_ind]
        weights_squared_sum += w * w
    return 0.5 * alpha * n_samples * wscale * wscale * weights_squared_sum 

cdef void _binomial_loss_regularized_all_samples64(
        SequentialDataset64 dataset, double wscale,
        double* weights,
        double* intercept,
        int n_samples, int n_features, int n_classes,
        double alpha, double beta, double* losses, double* prediction) nogil:
    """Compute binomial loss across all samples.

    Used for testing purpose only.
    losses is an array of 3 entries: prediction loss, l1loss, l2loss.
    """
    cdef double *x_data_ptr = NULL
    cdef int *x_ind_ptr = NULL
    cdef int xnnz = -1
    cdef double y
    cdef double sample_weight

    cdef int i, j, class_ind, feature_ind
    cdef double sum_predloss = 0.0
    cdef double sum_l1loss = 0.0
    cdef double sum_l2loss = 0.0

    for i in range(n_samples):
        # get next sample on the dataset
        # y must be in {0, 1}.
        dataset.next(&x_data_ptr, &x_ind_ptr, &xnnz,
                     &y, &sample_weight)

        # prediction of the multinomial classifier for the sample
        predict_sample64(x_data_ptr, x_ind_ptr, xnnz, weights, wscale,
                       intercept, prediction, n_classes)

        # compute the loss for this sample, given the prediction
        # DO NOT shift y from {0, 1} to {-1, 1} to use this function. Already shifted.
        # y = (y - 0.5) * 2
        sum_predloss += log(1.0 + exp(-1 * prediction[0] * y))
    sum_l1loss += _binomial_loss_l1_64(x_ind_ptr, xnnz, weights, wscale, beta, n_samples) 
    sum_l2loss += _binomial_loss_l2_64(x_ind_ptr, xnnz, weights, wscale, alpha, n_samples) 

    # Dividing by n_samples here to have per sample adjusted loss.
    # Multiplication by n_samples could simply be avoided in l1 and l2 calculation
    #   however this is clearer as using alpha_scaled and beta_scaled.
    losses[0] = sum_predloss / n_samples
    losses[1] = sum_l1loss / n_samples
    losses[2] = sum_l2loss / n_samples

cdef void _binomial_loss_regularized_all_samples32(
        SequentialDataset32 dataset, float wscale,
        float* weights,
        float* intercept,
        int n_samples, int n_features, int n_classes,
        double alpha, double beta, float* losses, float* prediction) nogil:
    """Compute binomial loss across all samples.

    Used for testing purpose only.
    losses is an array of 3 entries: prediction loss, l1loss, l2loss.
    """
    cdef float *x_data_ptr = NULL
    cdef int *x_ind_ptr = NULL
    cdef int xnnz = -1
    cdef float y
    cdef float sample_weight

    cdef int i, j, class_ind, feature_ind
    cdef float sum_predloss = 0.0
    cdef float sum_l1loss = 0.0
    cdef float sum_l2loss = 0.0

    for i in range(n_samples):
        # get next sample on the dataset
        # y must be in {0, 1}.
        dataset.next(&x_data_ptr, &x_ind_ptr, &xnnz,
                     &y, &sample_weight)

        # prediction of the multinomial classifier for the sample
        predict_sample32(x_data_ptr, x_ind_ptr, xnnz, weights, wscale,
                       intercept, prediction, n_classes)

        # compute the loss for this sample, given the prediction
        # DO NOT shift y from {0, 1} to {-1, 1} to use this function. Already shifted.
        # y = (y - 0.5) * 2
        sum_predloss += log(1.0 + exp(-1 * prediction[0] * y))
    sum_l1loss += _binomial_loss_l1_32(x_ind_ptr, xnnz, weights, wscale, beta, n_samples) 
    sum_l2loss += _binomial_loss_l2_32(x_ind_ptr, xnnz, weights, wscale, alpha, n_samples) 

    # Dividing by n_samples here to have per sample adjusted loss.
    # Multiplication by n_samples could simply be avoided in l1 and l2 calculation
    #   however this is clearer.
    losses[0] = sum_predloss / n_samples
    losses[1] = sum_l1loss / n_samples
    losses[2] = sum_l2loss / n_samples
