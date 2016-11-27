(ns elm.dataset
  (:require
    [clojure.data.csv :as csv]
    [clojure.java.io  :as io]))

(def read-resource-csv
  (comp csv/read-csv io/reader io/resource))

(def to-double #(Double/valueOf %))

(def explode (juxt #(map first %) #(map second %)))
(def implode #(apply map vector %&))

(defn boston
  []
  (->> "housing.data" read-resource-csv
       (map (comp #(split-at 13 %)
                  #(map to-double %)))))

(defn shuffle-split
  [ds & {:keys [n-splits test-size] :or {n-splits 1, test-size 0.05}}]
  (let [m        (count ds)
        test-num (* m test-size)]
    (map
      (fn [_] (split-at test-num (shuffle ds)))
      (range n-splits))))
