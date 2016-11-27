(ns elm.ejml
  (:require
    [clojure.core.matrix :as m])
  (:import
    [org.ejml.data DenseMatrix64F]
    [org.ejml.ops  CommonOps]))

(defn- to-dense-matrix
  [m]
  (->> m (map double-array) into-array (DenseMatrix64F.)))

(defn- from-dense-matrix
  [d]
  (let [n-col (.getNumCols d)]
    (m/matrix (partition n-col (seq (.getData d))))))

(defn pinv
  [m]
  (let [[n-row n-col] (m/shape m)
        minv (DenseMatrix64F. n-row n-col)]
    (CommonOps/pinv (to-dense-matrix m) minv)
    (from-dense-matrix minv)))

