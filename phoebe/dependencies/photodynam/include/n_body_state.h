#ifndef _N_BODY_STATE_
#define _N_BODY_STATE_

/*
  n_body_state.h

  Josh Carter, 2013

  Class defining NBodyState object.  Calls evolve described in n_body.h and coded in n_body.cpp  

*/

#include <cmath>
#include <limits>

#include "n_body_state.h"
#include "n_body.h"
#include "kepcart.h"

class NBodyState {
  
 private:
  double * mass;
  double * eta;
  double ** rj; //check on this (jacobian)
  double ** vj; 
  double ** aj;
  double ** rb; // barycentric
  double ** vb; 
  double ** rb_lt; // barycentric, lite-time corrected (observer at positive z)
  double time;
  int status;
  int N;
  
  void calcEta();
  
  void bary_coords(); 
  double bary_coords_lt(); 
  void calc_jac(double * a,double * e,double * in,double * o,double * ln,double * m); 

 public:
  // Constructor 1: initialize NBodyState with Jacobian coordinates
  NBodyState(double * m, double posj[][3], double velj[][3], int NN, double t0);
  // Constructor 2: initialize NBodyState with osculating elements for Jacobian coordinates.
  NBodyState(double * ms, double * a, double * e, double * in, double * o,double * ln, double * m, int NN, double t0);
 
  // Evolution overloaded operator.  t is time to evolve to, H is step size in BS integrator, ORBIT_ERROR is
  // orbit error tolerance and HLIMIT is minimum step size.
  // Example, for initialized state NBodyState state;
  //
  // state(100,1e-8,1e-16,1e-9); //evolves state to t = 0.
  double operator() (double t, double H, double ORBIT_ERROR, double HLIMIT);

  // Cruise operator moves according to velocity and acceleration and time t.  Returns new NBodyState.
  NBodyState * cruise(double t); //v,a time not reset

  // Returns current osculating keplerian elements for state
  void kep_elements(double * mj, double * a, double * e, double * in, double * o,double * ln, double * m); // return instantaneous keplerian elements

  // Returns mass of object obj
  double getMass(int obj);

  // Gets current time
  double getTime();

  // Gets the number of bodies in the state.
  int getN();

  // Returns NX3 matrix of positions for N objects corrected for the finite speed of light.
  double ** getBaryLT();

  // Returns barycentric coordinates for object obj.
  double X_B(int obj);
  double Y_B(int obj);
  double Z_B(int obj);

  // Returns barycentric velocities for object obj.
  double V_X_B(int obj);
  double V_Y_B(int obj);
  double V_Z_B(int obj);

  // Returns Jacobian coordinates for object obj.
  double X_J(int obj);
  double Y_J(int obj);
  double Z_J(int obj);

  // Returns Jacobian velocities for object obj.
  double V_X_J(int obj);
  double V_Y_J(int obj);
  double V_Z_J(int obj);

  // Returns barycentric, light-time corrected coordinates for object obj.
  double X_LT(int obj);
  double Y_LT(int obj);
  double Z_LT(int obj);

  // Returns barycentric, light-time corrected velocities for object obj.
  double V_X_LT(int obj);
  double V_Y_LT(int obj);
  double V_Z_LT(int obj);

  // Returns energy
  double getE();

  // Returns components of total angular momentum
  void getL(double * lx, double * ly, double * lz);
  
  // Destroys the state.
  ~NBodyState();

};

#define COPYN(a,b) for (int LOOP = 0; LOOP < N; LOOP++) {b[LOOP] = a[LOOP];}
#define COPYN3(a,b) for (int LOOP = 0; LOOP < N; LOOP++) for (int LOOP2 = 0; LOOP2 < 3; LOOP2++) {b[LOOP][LOOP2] = a[LOOP][LOOP2];}

#define CC (173.144483)

#define A3(r2,r1) for (int LOOP = 0; LOOP < 3; LOOP++) { r2[LOOP] = r1[LOOP]; }
#define A3P(r2,r1) for (int LOOP = 0; LOOP < 3; LOOP++) { r2[LOOP] += r1[LOOP];}
#define NORM2(r) (r[0]*r[0]+r[1]*r[1]+r[2]*r[2])
#define NORM_DIFF(r2,r1) (sqrt((r1[0]-r2[0])*(r1[0]-r2[0])+(r1[1]-r2[1])*(r1[1]-r2[1])+(r1[2]-r2[2])*(r1[2]-r2[2]))) 


NBodyState::NBodyState(double * m, double posj[][3], double velj[][3], int NN,double t0) {
  N = NN;
  mass = new double[N];
  eta = new double[N];

  rj = new double*[N];
  vj = new double*[N];
  aj = new double*[N];
  
  rb = new double*[N];
  vb = new double*[N];

  rb_lt = new double*[N];

  for (int i = 0; i < N; i++) {
    rj[i] = new double[3];
    vj[i] = new double[3];
    aj[i] = new double[3];

    rb[i] = new double[3];
    vb[i] = new double[3];

    rb_lt[i] = new double[3];
  }

  COPYN(m,mass);
  COPYN3(posj,rj);
  COPYN3(velj,vj);

  calcEta();

  evolve(rj,vj,aj,mass,eta,N,t0,t0,1,status,0,0);
 
  bary_coords();
  bary_coords_lt();

  time = t0;
}



NBodyState::NBodyState(double * ms, double * a, double * e, double * in, double * o,double * ln, double * m, int NN, double t0) {
  N = NN;
  mass = new double[N];
  eta = new double[N];

  rj = new double*[N];
  vj = new double*[N];
  aj = new double*[N];
  
  rb = new double*[N];
  vb = new double*[N];

  rb_lt = new double*[N];

  for (int i = 0; i < N; i++) {
    rj[i] = new double[3];
    vj[i] = new double[3];
    aj[i] = new double[3];

    rb[i] = new double[3];
    vb[i] = new double[3];

    rb_lt[i] = new double[3];
  }

  COPYN(ms,mass);

  calcEta();
  
  calc_jac(a,e,in,o,ln,m);  
  evolve(rj,vj,aj,mass,eta,N,t0,t0,1,status,0,0);
  bary_coords();
  bary_coords_lt();

  time = t0;
}

NBodyState::~NBodyState() {
  delete[] mass;
  delete[] eta;
  
  for (int i = 0; i < N; i++) {
    delete[] rj[i];
    delete[] vj[i];
    delete[] aj[i];
    delete[] rb[i];
    delete[] vb[i];
    delete[] rb_lt[i];
  }

  delete[] rj;
  delete[] vj;
  delete[] aj;
  delete[] rb;
  delete[] vb;
  delete[] rb_lt;
}

void NBodyState::calcEta() {
  eta[0] = mass[0];
  for (int i = 1; i < N; i++) eta[i] = eta[i-1]+mass[i];
}

double NBodyState::bary_coords_lt() {
  double dtMax= 0;
  // Linear Light Time Correction
  for (int i = 0; i < N; i++) {
    A3(rb_lt[i],rb[i]);
    if (fabs(rb[i][2]/CC) > dtMax) dtMax = fabs(rb[i][2]/CC);
    A3P(rb_lt[i],rb[i][2]/CC*vb[i]);
  }

  return dtMax;
}

void NBodyState::bary_coords() {
  A3(rb[0],rj[0]);
  A3(vb[0],vj[0]);
  for (int j = 1; j <= N-1; j++) {
    A3P(rb[0],-mass[j]/eta[j]*rj[j]);
    A3P(vb[0],-mass[j]/eta[j]*vj[j]);
  }

  for (int i = 1; i < N-1; i++) {
    A3(rb[i],rj[0]);
    A3(vb[i],vj[0]);
    A3P(rb[i],eta[i-1]/eta[i]*rj[i]);
    A3P(vb[i],eta[i-1]/eta[i]*vj[i]);  
    for (int j=i+1; j <= N-1; j++) {
      A3P(rb[i],-mass[j]/eta[j]*rj[j]);
      A3P(vb[i],-mass[j]/eta[j]*vj[j]);
    }
  }
  
  A3(rb[N-1],rj[0]);
  A3(vb[N-1],vj[0]);
  A3P(rb[N-1],eta[N-2]/eta[N-1]*rj[N-1]);
  A3P(vb[N-1],eta[N-2]/eta[N-1]*vj[N-1]);
}

void NBodyState::calc_jac(double * a, double * e, double * in, double * o, double * ln, double * m) {
  State kep_state;
  int count;

  // Set COM to origin and motionless
  for (int i = 0; i < 3; i++) {
    rj[0][i] = 0;
    vj[0][i] = 0;
  }

  for (int i = 1; i < N; i++) {
    cartesian(eta[i],a[i-1],e[i-1],in[i-1],ln[i-1],o[i-1],m[i-1],&kep_state,&count);
    rj[i][0] = kep_state.x;
    rj[i][1] = kep_state.y;
    rj[i][2] = kep_state.z;

    vj[i][0] = kep_state.xd;
    vj[i][1] = kep_state.yd;
    vj[i][2] = kep_state.zd;
  }
}

//Evolution operator -> () overloaded. t is integration time, H is max step.
double NBodyState::operator() (double t, double H, double ORBIT_ERROR, double HLIMIT) {
  
  evolve(rj,vj,aj,mass,eta,N,time,t,H,status,ORBIT_ERROR,HLIMIT); 
  time = t;
  bary_coords();
  bary_coords_lt();

  return status;
}

NBodyState * NBodyState::cruise(double t) {

  // simple... rj(t) = rj(time)+vj(time)*(t-time)+aj(time)/2*(t-time)^2

  double dt = t-time;
  double rn[N][3],vn[N][3];
  
  for (int i = 0; i < N; i++) {
    rn[i][0] = rj[i][0]+vj[i][0]*dt;//+aj[i][0]*dt*dt/2;
    rn[i][1] = rj[i][1]+vj[i][1]*dt;//+aj[i][1]*dt*dt/2;
    rn[i][2] = rj[i][2]+vj[i][2]*dt;//+aj[i][2]*dt*dt/2;
    vn[i][0] = vj[i][0];
    vn[i][1] = vj[i][1];
    vn[i][2] = vj[i][2];
  }

  NBodyState * st = new NBodyState(mass,rn,vn,N,t);

  return st;
} 			     

void NBodyState::kep_elements(double * mj, double * a, double * e, double * in, double * o, double * ln, double * m) {
  
  State kep_state;
  
  double a_,e_,in_,o_,ln_,m_;
  
  for (int i = 1; i < N; i++) {
    kep_state.x = rj[i][0];
    kep_state.y = rj[i][1];
    kep_state.z = rj[i][2];
    
    kep_state.xd = vj[i][0];
    kep_state.yd = vj[i][1];
    kep_state.zd = vj[i][2];
    
    keplerian(eta[i],kep_state,&a_,&e_,&in_,&ln_,&o_,&m_);
    mj[i-1] = eta[i];
    a[i-1] = a_;
    e[i-1] = e_;
    in[i-1] = in_;
    o[i-1] = o_;
    ln[i-1] = ln_;
    m[i-1] = m_;
  }
}

double NBodyState::getMass(int obj) { return mass[obj]; }
double NBodyState::getTime() { return time; }
int NBodyState::getN() { return N;}

double ** NBodyState::getBaryLT() { return rb_lt; } //Careful!!!

double NBodyState::X_B(int obj) { return rb[obj][0];} 
double NBodyState::Y_B(int obj) { return rb[obj][1];}
double NBodyState::Z_B(int obj) {return rb[obj][2];}

double NBodyState::V_X_B(int obj) {return vb[obj][0];}
double NBodyState::V_Y_B(int obj) {return vb[obj][1];}
double NBodyState::V_Z_B(int obj) {return vb[obj][2];}

double NBodyState::X_J(int obj) {return rj[obj][0];}
double NBodyState::Y_J(int obj) {return rj[obj][1];}
double NBodyState::Z_J(int obj) {return rj[obj][2];}

double NBodyState::V_X_J(int obj) {return vj[obj][0];}
double NBodyState::V_Y_J(int obj) {return vj[obj][1];}
double NBodyState::V_Z_J(int obj) {return vj[obj][2];}

double NBodyState::X_LT(int obj) {return rb_lt[obj][0];}
double NBodyState::Y_LT(int obj) {return rb_lt[obj][1];}
double NBodyState::Z_LT(int obj) {return rb_lt[obj][2];}

double NBodyState::V_X_LT(int obj) {return vb[obj][0];}
double NBodyState::V_Y_LT(int obj) {return vb[obj][1];}
double NBodyState::V_Z_LT(int obj) {return vb[obj][2];}

double NBodyState::getE() {
  double energy = 0;
  for (int j = 0; j < N; j++) {
    energy += mass[j]/mass[0]*NORM2(vb[j])/2;
    for (int i = 0; i < j; i++) energy -= mass[i]*mass[j]/mass[0]/NORM_DIFF(rb[i],rb[j]);
  }
  return energy;
}
  
void NBodyState::getL(double * lx, double * ly, double * lz) {
  *lx = 0; *ly = 0; *lz = 0;
  for (int i = 0; i < N; i++) {
    *lx += mass[i]*(rb[i][1]*vb[i][2]-rb[i][2]*vb[i][1]);
    *ly += mass[i]*(rb[i][2]*vb[i][0]-rb[i][0]*vb[i][2]);
    *lz += mass[i]*(rb[i][0]*vb[i][1]-rb[i][1]*vb[i][0]);
  }
}

#undef COPYN
#undef COPYN3
#undef CC

#undef A3
#undef A3P
#undef NORM2
#undef NORM_DIFF 
#endif
