(ns elm.core-test
  (:require
    [clojure.test               :refer :all]
    [clojure.core.matrix        :as m]
    [clojure.core.matrix.random :as r]
    [elm.core                   :refer :all]
    [elm.dataset                :as dataset]))

(deftest normalize-test
  (is (= [[[0.0 0.0 0.0] [4]]
          [[1.0 1.0 1.0] [7]]]
        (normalize [[[1 2 3] [4]]
                    [[4 5 6] [7]]]))))

(deftest forward-test
  (let [x (r/sample-normal [10 5])
        a (r/sample-normal [100 5])
        b (r/sample-normal 100)]
    (is (= [10 100]
           (m/shape (forward {:a a :b b :x x}))))))

(defn- mean [coll] (/ (reduce + coll) (count coll)))
(deftest boston-test
  (let [data   (normalize (dataset/boston))
        scores (map (fn [[test train]]
                      (cross-validation test train :L 100))
                    (dataset/shuffle-split data :n-splits 20))]
    (is (> (mean scores) 0.7))))
