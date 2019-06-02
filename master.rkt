#lang racket

(require 2htdp/batch-io)

(require "decision_functions.rkt")

;input dataset
(provide toytrain)
(define toytrain "../testcases/_mm_r1.csv")

(provide titanictrain)
(define titanictrain "../data/titanic_train.csv")

(provide mushroomtrain)
(define mushroomtrain "../data/mushrooms_train.csv")

;output tree (dot file)
(provide toyout)
(define toyout "../output/toy-decision-tree.dot")

;reading input datasets
;read the csv file myfile as a list of strings
;with each line of the original file as an element of the list
;further split each line at commas
;so then we have a list of list of strings
(provide toy-raw)
(define toy-raw (cdr (read-csv-file toytrain)))

(provide titanic-raw)
(define titanic-raw (map (lambda (x) (cddr x)) (cdr (read-csv-file titanictrain))))

(provide mushroom-raw)
(define mushroom-raw (cdr (read-csv-file mushroomtrain)))

;function to convert data to internal numerical format
;(features . result)
(provide format)
(define (format data)
  (define (operate l)
    (cond [(null? l) '()]
          [else (cons (string->number (car l)) (operate (cdr l)))]))
  (cons (operate (cdr data)) (string->number (car data))))

;list of (features . result)
(provide toy)
(define toy (map format toy-raw))

(provide titanic)
(define titanic (map format titanic-raw))

(provide mushroom)
(define mushroom (map format mushroom-raw))

;============================================================================================================
;============================================================================================================
;============================================================================================================

;get fraction of result fields that are 1
;used to find probability value at leaf
(provide get-leaf-prob)
(define (get-leaf-prob data)
  (define (ones data acc)
    (cond [(null? data) acc]
          [(= 1 (cdar data)) (ones (cdr data) (+ acc 1))]
          [else (ones (cdr data) acc)]))
  (/ (ones data 0) (length data))
  )

;;;;;;;;;;;;;;;;;;;;;;;;
;h definition;
(define (h x)
  (if (or (= x 0) (= x 1)) 0
    (* -1 (+ (* x (log x 2)) (* (- 1 x) (log (- 1 x) 2))))))
;;;;;;;;;;;;;;;;;;;;;;;;
;get entropy of dataset
(provide get-entropy)
(define (get-entropy data)
  (h (get-leaf-prob data)))



;find the difference in entropy achieved
;by applying a decision function f to the data
(provide entropy-diff)
(define (entropy-diff f data)
  (define tot (length data))
  (define (op x)
    (* (/ (length x) tot) (get-entropy x)))
  (define (eh l)
    (apply + (map op l)))
  (- (get-entropy data) (eh (filter1 f data '())))
  )

;choose the decision function that most reduces entropy of the data
(provide choose-f)
(define (choose-f candidates data) ; returns a decision function
  (argmax (lambda (x) (entropy-diff (cdr x) data)) candidates))

(provide DTree)
(struct DTree (desc func kids))

(define (filter1 f data acc)
    (define (find-and-fit p l)
    (cond [(null? l) (append l (list (list p)))]
          [(= (f (car p)) (f (caaar l))) (cons (cons p (car l)) (cdr l))]
          [else (cons (car l) (find-and-fit p (cdr l)))]))
    (cond [(null? data) acc]
          [else (filter1 f (cdr data) (find-and-fit (car data) acc))]))

;build a decision tree (depth limited) from the candidate decision functions and data
(provide build-tree)
(define (build-tree candidates data depth)
  (define prob-leaf (get-leaf-prob data))
  (cond [(or (null? candidates) (= depth 0) (= 1 prob-leaf) (= 0 prob-leaf)) (DTree (~a prob-leaf) get-leaf-prob '())]
        [else (define decider (choose-f candidates data))
              (define divided-data-list (filter1 (cdr decider) data '()))
              (define list-of-vals (map (lambda (x) ((cdr decider) (caar x))) divided-data-list))
              (DTree (car decider) (cons list-of-vals (cdr decider)) 
                     (map (lambda (x) (build-tree (remove decider candidates) x (- depth 1)))
                            divided-data-list))]))



;given a test data (features only), make a decision according to a decision tree
;returns probability of the test data being classified as 1
(provide make-decision)
(define (make-decision tree test)
  (match tree
    [(DTree desc func '()) (string->number desc)]
    [(DTree desc func subtrees-list)
     (if (pair? (member ((cdr func) test) (car func)))
         (make-decision (list-ref subtrees-list (index-of (car func) ((cdr func) test))) test) 0)]))

;============================================================================================================
;============================================================================================================
;============================================================================================================

;annotate list with indices
(define (pair-idx lst n)
  (if (empty? lst) `() (cons (cons (car lst) n) (pair-idx (cdr lst) (+ n 1))))
  )

;generate tree edges (parent to child) and recurse to generate sub trees
(define (dot-child children prefix tabs)
  (apply string-append
         (map (lambda (t)
                (string-append tabs
                               "r" prefix
                               "--"
                               "r" prefix "t" (~a (cdr t))
                               "[label=\"" (~a (cdr t)) "\"];" "\n"
                               (dot-helper (car t)
                                           (string-append prefix "t" (~a (cdr t)))
                                           (string-append tabs "\t")
                                           )
                               )
                ) children
                  )
         )
  )

;generate tree nodes and call function to generate edges
(define (dot-helper tree prefix tabs)
  (let* ([node (match tree [(DTree d f c) (cons d c)])]
         [d (car node)]
         [c (cdr node)])
    (string-append tabs
                   "r"
                   prefix
                   "[label=\"" d "\"];" "\n\n"
                   (dot-child (pair-idx c 0) prefix tabs)
                   )
    )
  )

;output tree (dot file)
(provide display-tree)
(define (display-tree tree outfile)
  (write-file outfile (string-append "graph \"decision-tree\" {" "\n"
                                     (dot-helper tree "" "\t")
                                     "}"
                                     )
              )
  )
;============================================================================================================
;============================================================================================================
;============================================================================================================
