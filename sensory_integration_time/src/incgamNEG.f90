  MODULE IncgamNEG
  USE set_precision
  IMPLICIT NONE
  PRIVATE
  PUBLIC  :: gseries, gexpan
  CONTAINS

      SUBROUTINE gseries(a,x,igam,ierr) BIND (c, NAME='gseries')
      USE Someconstants
      USE GammaError
      IMPLICIT NONE
      REAL(r8) :: a, x, igam
      REAL(r8) :: eps, p, q, t, v
      REAL(r8) :: logamma
      INTEGER ::  ierr, k, m
      eps=epss            
      t=1.0_r8/a;
      v=t;
      k=0
      m=0
      DO WHILE ((abs(t/v)>eps).AND.(m==0))
        p=(a+k)/(a+k+1);
        q=k+1;
        t=-x*t*p/q;
        v=v+t
        k=k+1
        IF (t>giant) m=1
      ENDDO
      IF (m==0) THEN
        IF (a>0.0_r8) THEN
          logamma=loggam(a)
          IF (logamma<loggiant) THEN
            igam=v/gamma(a)
          ELSE
            igam=0.0_r8
            ierr=1
          ENDIF
        ELSE
          IF (1-a<170) THEN
            igam=v/gamma(a)
          ELSE
            igam=0.0_r8
            ierr=1
          ENDIF
        ENDIF
      ELSE
        igam=0.0_r8
        ierr=1
      ENDIF
      END SUBROUTINE gseries

      SUBROUTINE gexpan(a,x,igam,ierr) BIND (c, NAME='gexpan')
      USE Someconstants
      USE GammaError
      IMPLICIT NONE
      REAL(r8) :: a, x, igam
      REAL(r8) :: eps, p, t, v
      INTEGER ::  ierr, k, m
      eps=epss            
      ierr=0
      t=1.0_r8;
      v=t;
      k=0
      m=0
      DO WHILE ((abs(t/v)>eps).AND.(m==0))
        p=(1.0_r8-a+k);
        t=t*p/x;
        v=v+t
        k=k+1
        IF (t>giant) m=1
      ENDDO
      IF (m==0) THEN
        IF ((x>loggiant).OR.(loggam(a)>loggiant)) THEN
          m=1
        ELSE 
          igam=v*exp(x)/(x*gamma(a));
        ENDIF
      ENDIF
      IF (m==1) THEN
        igam=0.0_r8
        ierr=1
      ENDIF
      END SUBROUTINE gexpan

      FUNCTION exmin1(x,eps)
      USE Someconstants  
      IMPLICIT NONE
      !computes (exp(x)-1)/x 
      REAL(r8) :: exmin1, x, eps
      REAL(r8) :: t, y
      IF (x==0) THEN
        y=1.0_r8
      ELSEIF ((x<-0.69_r8).OR.(x>0.4_r8)) THEN
        y=(exp(x)-1.0_r8)/x
      ELSE
        t=x*0.5_r8;
        y=exp(t)*sinh(t,eps)/t
      ENDIF
      exmin1=y
      END FUNCTION exmin1
      RECURSIVE FUNCTION sinh(x,eps) RESULT(sinhh)
      USE Someconstants  
      IMPLICIT NONE
      !to compute hyperbolic function sinh (x)}
      REAL(r8) :: sinhh, x, eps
      REAL(r8) :: ax, e, t, x2, y
      INTEGER  :: u, k
      ax=abs(x);
      IF (x==0.0_r8) THEN
        y=0.0_r8
      ELSEIF (ax<0.12) THEN
        e=eps*0.1_r8;
        x2=x*x;
        y=1;
        t=1;
        u=0;
        k=1;
        DO WHILE(t>e)
          u=u+8*k-2;
          k=k+1;
          t=t*x2/u;
          y=y+t
        END DO
        y=x*y
      ELSEIF (ax<0.36_r8) THEN
        t=sinh(x*0.333333333333333333333333333333_r8,eps);
        y=t*(3.0_r8+4.0_r8*t*t);
      ELSE
        t=exp(x);
        y=(t-1.0_r8/t)*0.5_r8
      ENDIF
      sinhh=y
      END FUNCTION sinh
   END MODULE IncgamNEG


 
