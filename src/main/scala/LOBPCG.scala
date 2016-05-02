import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.Vectors
import breeze.linalg._
import org.apache.spark.mllib.linalg.{DenseMatrix => DM}
import scala.util.control.Breaks._

object LOBPCG {
  def lobpcg(A:IndexedRowMatrix, k:Int, largest:Boolean = true, maxiter:Int=20, tol:Double=10e-6):(Array[Double],Matrix) = {
	  def IndexedRowMatrixToDenseMatrix(M:IndexedRowMatrix):DenseMatrix[Double] ={
		val m = M.rows.collect()
		val a = DenseMatrix.zeros[Double](m.size,m(0).vector.size)
		for(i <- 0 until a.rows) {
			val v= m(i).vector.toArray
			for(j <- 0 until a.cols)
				a(i,j) = v(j)
		}
		a
	  }
	  def matrixNorm(w:DenseMatrix[Double]):Double = math.sqrt(sum(w.t * w))
	  val n = A.rows.count().toInt
	  var X = DenseMatrix.rand(n,k)
	  var P = X
	  var lambda:DenseVector[Double]=null
	  breakable {
	  	for(i <- 0 until maxiter){
	  		val Ax = IndexedRowMatrixToDenseMatrix(A.multiply(new DM(X.rows,X.cols,X.data)))
	  		val xAx = X.t*Ax
			lambda = diag(xAx) :/ diag(X.t * X )
			val W = X*diag(lambda) - Ax
			val wnorm = 
			if (matrixNorm(W) <= tol) break
			val R = qr.justQ(if (i ==0) DenseMatrix.horzcat(X,W) else DenseMatrix.horzcat(X,W,P))
			val Rsystem = R.t* IndexedRowMatrixToDenseMatrix(A.multiply(new DM(R.rows,R.cols,R.data)))
			val eig.Eig(_,_,nX) = eig(Rsystem)
			P = X
			X = R * ( if (!largest) nX(::, (n-k) until n) else nX(::,0 until k) )
	  	}
	  }
	(lambda.toArray,new DM(X.rows,X.cols,X.data))
  }
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("LOBPCG APP")
    val sc = new SparkContext(conf)
    val n = 30
    val k = 3
    var A = DenseMatrix.rand(n,n)
    A = A.t * A / 2.0
    val q = for(i<-0 until A.cols) yield A(::,i).toArray

    val M = new IndexedRowMatrix( sc.parallelize(q).zipWithIndex.map({case (v,i) => new IndexedRow(i,Vectors.dense(v))}))
    val (x,r) = lobpcg(M,k)
    val eig.Eig(vals,_,vecs) = eig(A)
    //println(vals((n-k) until n))
    val svd = M.computeSVD(k)
    println(Vectors.dense(vals( 0 until k).toArray))
    println(svd.s)
    println(Vectors.dense(x))
//    x.foreach(println)
    //println(vecs)


  }
}
