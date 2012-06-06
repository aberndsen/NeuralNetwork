!compile with
!f2py -c -m sigmoid2d --f90flags="-fopenmp -lm " --opt="-O3 " -lgomp --fcompiler=gnu95 sigmoid2d.f90 

subroutine sigmoid2d(z,nr,nc)
  use omp_lib
  implicit none
  real*8 z(nr,nc)
  integer nr, nc
  integer :: i, j, num_threads
!  integer OMP_get_num_threads
  !f2py intent(in) :: z
  !f2py intent(hide) :: nr, nc
  !f2py intent(out) :: z
  
!$OMP PARALLEL DO PRIVATE(i),shared(z)
  do i=1,nr 
!     if (j == 1) then
        ! The if clause can be removed for serious use.
        ! It is here for debugging only.
!        num_threads = OMP_get_num_threads()
!        print *, 'num_threads running:', num_threads
!     end if
     do j=1,nc
        z(i,j) = 1.d0/(1.d0 + exp(-1.d0*z(i,j)))
     end do
  end do
!$OMP END PARALLEL DO

end subroutine sigmoid2d

  
