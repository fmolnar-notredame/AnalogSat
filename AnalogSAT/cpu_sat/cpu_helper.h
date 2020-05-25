//This file is part of AnalogSAT
//Copyright(C) 2019 Ferenc Molnar
//
//AnalogSAT is free software: you can redistribute it and / or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.
//
//This program is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
//GNU General Public License for more details.
//
//You should have received a copy of the GNU General Public License
//along with this program. If not, see <http://www.gnu.org/licenses/>.


#ifndef ANALOGSAT_CPU_HELPER_H
#define ANALOGSAT_CPU_HELPER_H

namespace analogsat
{
	template<typename TFloat, int K, int Q>
	struct CpuProdWrapper { static inline TFloat KmiProd(TFloat *s); };

	template<typename TFloat> struct CpuProdWrapper<TFloat, 1, 1> { static inline TFloat KmiProd(TFloat *s) { return (TFloat)1; }; };

	template<typename TFloat> struct CpuProdWrapper<TFloat, 2, 1> { static inline TFloat KmiProd(TFloat *s) { return s[1]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 2, 2> { static inline TFloat KmiProd(TFloat *s) { return s[0]; }; };

	template<typename TFloat> struct CpuProdWrapper<TFloat, 3, 1> { static inline TFloat KmiProd(TFloat* s) { return s[1] * s[2]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 3, 2> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[2]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 3, 3> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1]; }; };

	template<typename TFloat> struct CpuProdWrapper<TFloat, 4, 1> { static inline TFloat KmiProd(TFloat* s) { return s[1] * s[2] * s[3]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 4, 2> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[2] * s[3]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 4, 3> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[3]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 4, 4> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2]; }; };

	template<typename TFloat> struct CpuProdWrapper<TFloat, 5, 1> { static inline TFloat KmiProd(TFloat* s) { return s[1] * s[2] * s[3] * s[4]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 5, 2> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[2] * s[3] * s[4]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 5, 3> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[3] * s[4]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 5, 4> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[4]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 5, 5> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3]; }; };

	template<typename TFloat> struct CpuProdWrapper<TFloat, 6, 1> { static inline TFloat KmiProd(TFloat* s) { return s[1] * s[2] * s[3] * s[4] * s[5]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 6, 2> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[2] * s[3] * s[4] * s[5]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 6, 3> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[3] * s[4] * s[5]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 6, 4> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[4] * s[5]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 6, 5> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[5]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 6, 6> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4]; }; };

	template<typename TFloat> struct CpuProdWrapper<TFloat, 7, 1> { static inline TFloat KmiProd(TFloat* s) { return s[1] * s[2] * s[3] * s[4] * s[5] * s[6]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 7, 2> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[2] * s[3] * s[4] * s[5] * s[6]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 7, 3> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[3] * s[4] * s[5] * s[6]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 7, 4> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[4] * s[5] * s[6]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 7, 5> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[5] * s[6]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 7, 6> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[6]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 7, 7> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[5]; }; };

	template<typename TFloat> struct CpuProdWrapper<TFloat, 8, 1> { static inline TFloat KmiProd(TFloat* s) { return s[1] * s[2] * s[3] * s[4] * s[5] * s[6] * s[7]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 8, 2> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[2] * s[3] * s[4] * s[5] * s[6] * s[7]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 8, 3> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[3] * s[4] * s[5] * s[6] * s[7]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 8, 4> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[4] * s[5] * s[6] * s[7]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 8, 5> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[5] * s[6] * s[7]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 8, 6> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[6] * s[7]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 8, 7> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[5] * s[7]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 8, 8> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[5] * s[6]; }; };

	template<typename TFloat> struct CpuProdWrapper<TFloat, 9, 1> { static inline TFloat KmiProd(TFloat* s) { return s[1] * s[2] * s[3] * s[4] * s[5] * s[6] * s[7] * s[8]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 9, 2> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[2] * s[3] * s[4] * s[5] * s[6] * s[7] * s[8]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 9, 3> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[3] * s[4] * s[5] * s[6] * s[7] * s[8]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 9, 4> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[4] * s[5] * s[6] * s[7] * s[8]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 9, 5> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[5] * s[6] * s[7] * s[8]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 9, 6> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[6] * s[7] * s[8]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 9, 7> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[5] * s[7] * s[8]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 9, 8> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[5] * s[6] * s[8]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 9, 9> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[5] * s[6] * s[7]; }; };

	template<typename TFloat> struct CpuProdWrapper<TFloat, 10, 1 > { static inline TFloat KmiProd(TFloat* s) { return s[1] * s[2] * s[3] * s[4] * s[5] * s[6] * s[7] * s[8] * s[9]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 10, 2 > { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[2] * s[3] * s[4] * s[5] * s[6] * s[7] * s[8] * s[9]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 10, 3 > { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[3] * s[4] * s[5] * s[6] * s[7] * s[8] * s[9]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 10, 4 > { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[4] * s[5] * s[6] * s[7] * s[8] * s[9]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 10, 5 > { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[5] * s[6] * s[7] * s[8] * s[9]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 10, 6 > { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[6] * s[7] * s[8] * s[9]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 10, 7 > { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[5] * s[7] * s[8] * s[9]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 10, 8 > { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[5] * s[6] * s[8] * s[9]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 10, 9 > { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[5] * s[6] * s[7] * s[9]; }; };
	template<typename TFloat> struct CpuProdWrapper<TFloat, 10, 10> { static inline TFloat KmiProd(TFloat* s) { return s[0] * s[1] * s[2] * s[3] * s[4] * s[5] * s[6] * s[7] * s[8]; }; };

	// for each literal the RHS summation is repeated. This loop can be unrolled, but
	// the compiler needs to insert the loop argument into a template argument
	// solution: template metaprogramming, compile-time loop via template argument	

	template<typename TFloat>
	inline TFloat sigmoid(TFloat x)
	{
		return x / sqrt((TFloat)1 + x * x);
	}
	
	template<typename TFloat, int K, int Q>
	struct CpuKmiWrapper
	{
		static inline void KmiFunc(TFloat* rhs, int* index, TFloat* s, TFloat r)
		{
			if (index[Q] != 0)
			{
				TFloat rr = r * CpuProdWrapper<TFloat, K, Q + 1>::KmiProd(s);	// calculate r * Kmi
				rr *= index[Q] < 0 ? (TFloat)-2.0 : (TFloat)2.0;				//multiply by sign
				rhs[abs(index[Q])] += rr;										//accumulate RHS contribution
				CpuKmiWrapper<TFloat, K, Q + 1>::KmiFunc(rhs, index, s, r);		//template recursion
			}
		}

		static inline void KmiFuncTanh(TFloat* rhs, int* index, TFloat* s, TFloat r, TFloat q)
		{
			if (index[Q] != 0)
			{
				TFloat kmi = CpuProdWrapper<TFloat, K, Q + 1>::KmiProd(s);
				TFloat rr = r * (kmi < 0 ? -kmi : kmi) * tanh(q * s[Q]); //knorm * am * |Kmi| * tanh(q * s[i])				
				rr *= index[Q] < 0 ? (TFloat)-1.0 : (TFloat)1.0; //c_mi (sign)
				rhs[abs(index[Q])] += rr;
				CpuKmiWrapper<TFloat, K, Q + 1>::KmiFuncTanh(rhs, index, s, r, q);
			}
		}
	};


	// compile-time loop termination
	template<typename TFloat> struct CpuKmiWrapper<TFloat, 1, 1> 
	{ 
		static inline void KmiFunc(TFloat* rhs, int* index, TFloat* s, TFloat r){} 
		static inline void KmiFuncTanh(TFloat* rhs, int* index, TFloat* s, TFloat r, TFloat q){}
	};
	template<typename TFloat> struct CpuKmiWrapper<TFloat, 2, 2> 
	{ 
		static inline void KmiFunc(TFloat* rhs, int* index, TFloat* s, TFloat r){} 
		static inline void KmiFuncTanh(TFloat* rhs, int* index, TFloat* s, TFloat r, TFloat q){}
	};
	template<typename TFloat> struct CpuKmiWrapper<TFloat, 3, 3> 
	{ 
		static inline void KmiFunc(TFloat* rhs, int* index, TFloat* s, TFloat r){} 
		static inline void KmiFuncTanh(TFloat* rhs, int* index, TFloat* s, TFloat r, TFloat q){}
	};
	template<typename TFloat> struct CpuKmiWrapper<TFloat, 4, 4> 
	{ 
		static inline void KmiFunc(TFloat* rhs, int* index, TFloat* s, TFloat r){} 
		static inline void KmiFuncTanh(TFloat* rhs, int* index, TFloat* s, TFloat r, TFloat q){}
	};
	template<typename TFloat> struct CpuKmiWrapper<TFloat, 5, 5> 
	{ 
		static inline void KmiFunc(TFloat* rhs, int* index, TFloat* s, TFloat r){} 
		static inline void KmiFuncTanh(TFloat* rhs, int* index, TFloat* s, TFloat r, TFloat q){}
	};
	template<typename TFloat> struct CpuKmiWrapper<TFloat, 6, 6> 
	{ 
		static inline void KmiFunc(TFloat* rhs, int* index, TFloat* s, TFloat r){} 
		static inline void KmiFuncTanh(TFloat* rhs, int* index, TFloat* s, TFloat r, TFloat q){}
	};
	template<typename TFloat> struct CpuKmiWrapper<TFloat, 7, 7> 
	{ 
		static inline void KmiFunc(TFloat* rhs, int* index, TFloat* s, TFloat r){} 
		static inline void KmiFuncTanh(TFloat* rhs, int* index, TFloat* s, TFloat r, TFloat q){}
	};
	template<typename TFloat> struct CpuKmiWrapper<TFloat, 8, 8> 
	{ 
		static inline void KmiFunc(TFloat* rhs, int* index, TFloat* s, TFloat r){} 
		static inline void KmiFuncTanh(TFloat* rhs, int* index, TFloat* s, TFloat r, TFloat q){}
	};
	template<typename TFloat> struct CpuKmiWrapper<TFloat, 9, 9> 
	{ 
		static inline void KmiFunc(TFloat* rhs, int* index, TFloat* s, TFloat r){} 
		static inline void KmiFuncTanh(TFloat* rhs, int* index, TFloat* s, TFloat r, TFloat q){}
	};
	template<typename TFloat> struct CpuKmiWrapper<TFloat, 10, 10> 
	{ 
		static inline void KmiFunc(TFloat* rhs, int* index, TFloat* s, TFloat r){} 
		static inline void KmiFuncTanh(TFloat* rhs, int* index, TFloat* s, TFloat r, TFloat q){}
	};

}

#endif
