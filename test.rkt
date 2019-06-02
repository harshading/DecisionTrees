#lang racket

(require "master.rkt")
(require "decision_functions.rkt")
(require "testdata.rkt")

(define dotfile
  (display-tree (build-tree (list y1 y2 y3 y4>62) toy 2) toyout)
  )

(define (test1)
  (let* ([train toy]
         [test toy_test]
         [candidates (list y1 y2 y3 y4>62)]
         [dtree (build-tree candidates train 4)])
    (map (lambda (x) (make-decision dtree x)) test)
    )
  )

(define (test2)
  (let* ([train titanic]
         [test titanic_test]
         [candidates (list pclass sex age>25 sibsp parch fare>50 emb)]
         [dtree (build-tree candidates train 6)])
    (map (lambda (x) (make-decision dtree x)) test)
    )
  )

(define (test3)
  (let* ([train mushroom]
         [test mushroom_test]
         [candidates (list cshape csurf bruise odor gatch gspace gsize sshape nring pop hab)]
         [dtree (build-tree candidates train 11)])
    (map (lambda (x) (make-decision dtree x)) test)
    )
  )