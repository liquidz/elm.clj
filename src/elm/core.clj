(ns elm.core
  (:refer-clojure
    :exclude [* - + == / < <= > >= not= min max])
  (:require
    [clojure.core.matrix           :as m]
    [clojure.core.matrix.operators :refer :all]
    [clojure.core.matrix.random    :as r]
    [elm.dataset                   :as dataset]
    [elm.ejml                      :as ejml]))

(def max* #(apply max %))
(def min* #(apply min %))

(defn normalize
  [ds & {:keys [from to] :or {from 0.0, to 1.0}}]
  (let [[x y] (dataset/explode ds)
        xt    (m/transpose x)
        xmax  (map max* xt)
        xmin  (map min* xt)
        x-std (/ (- x xmin) (- xmax xmin))]
    (dataset/implode
      (+ (* x-std (clojure.core/- to from)) from)
      y)))

(defn sigmoid
  [x]
  (/ 1 (+ 1 (m/exp (- x)))))

(defn forward
  [{:keys [a b x]}]
  (sigmoid (+ (m/transpose (m/dot a (m/transpose x))) b)))

(defn fit
  [{:keys [x n-hidden] :as rgrsr} y]
  (let [[_ d] (m/shape x)
        a     (r/sample-normal [n-hidden d])
        b     (r/sample-normal n-hidden)
        rgrsr (assoc rgrsr :a a :b b)
        h     (forward rgrsr)]
    (assoc rgrsr :beta (m/dot (ejml/pinv h) y))))

(defn predict
  [{:keys [x beta] :as rgrsr}]
  (let [[n d] (m/shape x)
        h     (forward rgrsr)]
    (m/dot h beta)))

(defn score
  "Returns the coefficient of determination R^2 of the prediction.

  The coefficient R^2 is defined as (1 - u/v), where u is the regression
  sum of squares ((y_true - y_pred) ** 2).sum() and v is the residual sum
  of squares ((y_true - y_true.mean()) ** 2).sum().

  c.f. scikit-learn RegressorMixin
  "
  [rgrsr y]
  (let [pred  (predict rgrsr)
        y-mean (/ (reduce + y) (count y))
        u      (reduce + (** (- y pred) 2))
        v      (reduce + (** (- y y-mean) 2))]
    (first (- 1.0 (/ u v)))))

(defn regressor
  [x & {:keys [n-hidden a b beta]
        :or   {n-hidden 2000}
        :as   param}]
  (assoc param :x (m/matrix x)))

(defn cross-validation
  [test train & {:keys [L]}]
  (let [[x y]   (dataset/explode train)
        [tx ty] (dataset/explode test)
        res     (fit (regressor x :n-hidden L) y)]
    (score (assoc res :x tx) ty)))
