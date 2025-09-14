// sc_thermal.cpp — first‑order thermal RC + proportional fan
#include <systemc>
#include <fstream>
using namespace sc_core;

SC_MODULE(Thermal) {
  double T=40.0, Tamb=25.0, Rth=0.3, tau=20.0, fan=0.0;
  SC_CTOR(Thermal){ SC_THREAD(run); }
  void run(){
    std::ifstream fin("power.csv"); std::vector<double> P; double v;
    while(fin>>v) P.push_back(v);
    std::ofstream fout("thermal.csv"); double t=0.0;
    for(size_t i=0;i<P.size();++i){
      double Pnow=P[i];
      double target=75.0;
      fan = std::max(0.0, std::min(100.0, (T-target)*3.0));      // simple controller
      double dTdt = (Pnow*Rth - (T - Tamb)) / tau;               // RC model
      T += dTdt * 1.0;
      fout << t << "," << T << "," << fan << "\n";
      wait(sc_time(1, SC_SEC)); t += 1.0;
    }
    sc_stop();
  }
};
int sc_main(int,char**){ Thermal th("th"); sc_start(); return 0; }
