  MODULE Someconstants
  USE set_precision
  IMPLICIT NONE
    REAL(r8), PARAMETER :: lnsqrttwopi=0.9189385332046727418_r8     ! log(sqrt(2*pi))
    REAL(r8), PARAMETER :: sqrttwopi=2.5066282746310005024_r8       ! sqrt(2*pi)
    REAL(r8), PARAMETER :: eulmasc=0.5772156649015328606_r8         ! Euler-Mascheroni constant
    REAL(r8), PARAMETER :: pihalf=1.5707963267948966192_r8          ! pi/2
    REAL(r8), PARAMETER :: pikwart=0.7853981633974483096_r8         ! pi/4
    REAL(r8), PARAMETER :: pidriekwart=2.3561944901923449288_r8     ! 3pi/4
    REAL(r8), PARAMETER :: sqrtpi=1.7724538509055160272_r8          ! sqrt(pi)
    REAL(r8), PARAMETER :: lnpi=1.1447298858494001741_r8            ! log(pi)
    REAL(r8), PARAMETER :: pi=3.1415926535897932385_r8              ! pi
    REAL(r8), PARAMETER :: onepi=.31830988618379067153_r8           ! 1/pi
    REAL(r8), PARAMETER :: sqrt2=1.4142135623730950488_r8           ! sqrt(2)
    REAL(r8), PARAMETER :: sqrt3=1.7320508075688772935_r8           ! sqrt(3)
    REAL(r8), PARAMETER :: sqrt2opi=0.7978845608028653559_r8        ! sqrt(2/pi)
    REAL(r8), PARAMETER :: twopi=6.2831853071795864769_r8           ! 2*pi
    REAL(r8), PARAMETER :: oneoversqrtpi=0.5641895835477562869_r8   ! 1/sqrt(pi)
    REAL(r8), PARAMETER :: twooversqrtpi=1.1283791670955125739_r8   ! 2/sqrt(pi)
    REAL(r8), PARAMETER :: onethird=0.33333333333333333_r8          ! 1/3
    REAL(r8), PARAMETER :: twothird=0.6666666666666666666667_r8     ! 2/3
    REAL(r8), PARAMETER :: onesix=.166666666666666666666666666667_r8! 1/6 
    REAL(r8), PARAMETER :: piquart=0.78539816339744830962_r8        ! pi/4
    REAL(r8), PARAMETER :: twoexp14=1.18920711500272106671749997_r8 ! 2**(1/4)
    REAL(r8), PARAMETER :: epss=1.e-15_r8                           ! demanded accuracy 
    REAL(r8), PARAMETER :: dwarf=TINY(0.0_r8)*1000.0_r8             ! safe underflow limit
    REAL(r8), PARAMETER :: logdwarf=log(dwarf)                      ! safe log underflow limit
    REAL(r8), PARAMETER :: giant=HUGE(0.0_r8)/1000.0_r8             ! safe overflow limit
    REAL(r8), PARAMETER :: loggiant=log(giant)                      ! safe log overflow limit
    REAL(r8), PARAMETER :: giantsq=sqrt(giant)                      ! square root of giant
    REAL(r8), PARAMETER :: machtol=EPSILON(0.0_r8)                  ! machine-epsilon    
    REAL(r8), PARAMETER :: explow = -300;
    REAL(r8), PARAMETER :: exphigh = 300;
  END MODULE Someconstants
