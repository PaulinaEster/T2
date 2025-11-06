/* 
 * --------------------------------------------------------------------------------------------------------------
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 * As a special exception, you may use this file as part of a free software
 * library without restriction.  Specifically, if other files instantiate
 * templates or use macros or inline functions from this file, or you compile
 * this file and link it with other files to produce an executable, this
 * file does not by itself cause the resulting executable to be covered by
 * the GNU General Public License.  This exception does not however
 * invalidate any other reasons why the executable file might be covered by
 * the GNU General Public License.
 *
 * --------------------------------------------------------------------------------------------------------------
 * Authors: 
 *      Dalvan Griebler <dalvangriebler@gmail.com>
 *      Gabriell Araujo <hexenoften@gmail.com>
 * --------------------------------------------------------------------------------------------------------------
 */ 
#include "common_serial.h"

#define WORKLOAD_A
#if defined(WORKLOAD_A)
#define WORKLOAD "A"
#define WORKLOAD_CHECKSUM_VALUE_1 763840LL
#define WORKLOAD_CHECKSUM_VALUE_2 21700LL
#define WORKLOAD_CHECKSUM_VALUE_3 30256LL
#define WORKLOAD_CHECKSUM_VALUE_4 26846LL
#define N 32
#elif defined(WORKLOAD_B)
#define WORKLOAD "B"
#define WORKLOAD_CHECKSUM_VALUE_1 6308736LL 
#define WORKLOAD_CHECKSUM_VALUE_2 89460LL
#define WORKLOAD_CHECKSUM_VALUE_3 165984LL
#define WORKLOAD_CHECKSUM_VALUE_4 111006LL
#define N 64
#elif defined(WORKLOAD_C)
#define WORKLOAD "C"
#define WORKLOAD_CHECKSUM_VALUE_1 51271424LL  
#define WORKLOAD_CHECKSUM_VALUE_2 363220LL
#define WORKLOAD_CHECKSUM_VALUE_3 1016000LL 
#define WORKLOAD_CHECKSUM_VALUE_4 451358LL
#define N 128
#elif defined(WORKLOAD_D)
#define WORKLOAD "D"
#define WORKLOAD_CHECKSUM_VALUE_1 413396480LL
#define WORKLOAD_CHECKSUM_VALUE_2 1463700LL
#define WORKLOAD_CHECKSUM_VALUE_3 6865280LL
#define WORKLOAD_CHECKSUM_VALUE_4 1820190LL
#define N 256
#elif defined(WORKLOAD_E)
#define WORKLOAD "E"
#define WORKLOAD_CHECKSUM_VALUE_1 3320110080LL
#define WORKLOAD_CHECKSUM_VALUE_2 5876500LL
#define WORKLOAD_CHECKSUM_VALUE_3 49840896LL 
#define WORKLOAD_CHECKSUM_VALUE_4 7310366LL
#define N 512
#elif defined(WORKLOAD_F)
#define WORKLOAD "F"
#define WORKLOAD_CHECKSUM_VALUE_1 26612709376LL
#define WORKLOAD_CHECKSUM_VALUE_2 23549460LL
#define WORKLOAD_CHECKSUM_VALUE_3 378340864LL 
#define WORKLOAD_CHECKSUM_VALUE_4 29300766LL
#define N 1024
#elif defined(WORKLOAD_G)
#define WORKLOAD "G"
#define WORKLOAD_CHECKSUM_VALUE_1 213109141504LL   
#define WORKLOAD_CHECKSUM_VALUE_2 94284820LL
#define WORKLOAD_CHECKSUM_VALUE_3 2945059840LL   
#define WORKLOAD_CHECKSUM_VALUE_4 117321758LL
#define N 2048
#elif defined(WORKLOAD_H)
#define WORKLOAD "H"
#define WORKLOAD_CHECKSUM_VALUE_1 1705703301120LL
#define WORKLOAD_CHECKSUM_VALUE_2 377313300LL
#define WORKLOAD_CHECKSUM_VALUE_3 23233566720LL   
#define WORKLOAD_CHECKSUM_VALUE_4 469524510LL
#define N 4096
#else
#define WORKLOAD_A
#define WORKLOAD "A"
#define WORKLOAD_A_REFERENCE_VALUE_1 763840LL
#define WORKLOAD_A_REFERENCE_VALUE_2 21700LL
#define WORKLOAD_A_REFERENCE_VALUE_3 30256LL
#define WORKLOAD_A_REFERENCE_VALUE_4 26846LL
#define N 32
#endif

#define TIMER_TOTAL 0
#define TIMER_MEMORY_TRANSFERS 1
#define TIMER_LINEARIZATION 2
#define TIMER_COMPUTATION 3
#define TIMER_DESLINEARIZATION 4

// global variables
int passed_verification;

// matrix multiplication function prototype
// void matrix_multiplication(int** matrix1, int** matrix2, int** matrix3);

// other function prototypes
void initialization(int** matrix1, int** matrix2, int** matrix3);
void verification(int** matrix);
void debug_results(int** matrix);
void release_resources(int** matrix1, int** matrix2, int** matrix3);

void initialization(int** matrix1, int** matrix2, int** matrix3){
	// setup common stuff
	setup_common();	

	// initial values of the global arrays
	for(int i=0; i< N; i++){
		for(int j=0; j< N; j++){
			matrix1[i][j] = 4;
			matrix2[i][j] = 5;
			matrix3[i][j] = 0;

			if(i == j){
				matrix1[i][j] = i;
				matrix2[i][j] = j;
			}
		}
	}	
}

void verification(int** matrix){
	long long int value1 = 0;
	long long int value2 = 0;
	long long int value3 = 0;
	long long int value4 = 0;

	int i = 0;
	int j = 0;

	// total
	i = 0;
	j = 0;
	for(i=0; i<N; i++){
		for(j=0; j<N; j++){
			value1 += (long long int) matrix[i][j];
		}
	}

	// column 0
	i = 0;
	j = 0;
	for(i=0; i<N; i++){
		value2 += (long long int) matrix[i][0];
	}	

	// diagonal
	i = 0;
	j = 0;
	for(i=0; i<N; i++){
		value3 += (long long int) matrix[i][i];
	}

	// line N-1
	i = 0;
	j = 0;
	for(j=0; j<N; j++){
		value4 += (long long int) matrix[N-1][j];
	}

	if( (WORKLOAD_CHECKSUM_VALUE_1==value1) &&
			(WORKLOAD_CHECKSUM_VALUE_2==value2) &&
			(WORKLOAD_CHECKSUM_VALUE_3==value3) &&
			(WORKLOAD_CHECKSUM_VALUE_4==value4) ){
		passed_verification = 1;
	}
	else{
		passed_verification = 0;
	}

	char checksum_string_aux[256];	
	sprintf(checksum_string_aux, "%25s\t%20s\t%20s\n", "Reference", "Correct", "Found");
	strcpy(checksum_string, checksum_string_aux);
	sprintf(checksum_string_aux, "%25s\t%20lld\t%20lld\n", "checksum_1", WORKLOAD_CHECKSUM_VALUE_1, value1);
	strcat(checksum_string, checksum_string_aux);
	sprintf(checksum_string_aux, "%25s\t%20lld\t%20lld\n", "checksum_2", WORKLOAD_CHECKSUM_VALUE_2, value2);
	strcat(checksum_string, checksum_string_aux);
	sprintf(checksum_string_aux, "%25s\t%20lld\t%20lld\n", "checksum_3", WORKLOAD_CHECKSUM_VALUE_3, value3);
	strcat(checksum_string, checksum_string_aux);
	sprintf(checksum_string_aux, "%25s\t%20lld\t%20lld", "checksum_4", WORKLOAD_CHECKSUM_VALUE_4, value4);
	strcat(checksum_string, checksum_string_aux);

	char timer_string_aux[256];	
	sprintf(timer_string_aux, "%25s\t%20s\t%20s\n", "Timer", "Time (s)", "Percentage");
	strcpy(timer_string, timer_string_aux);
	sprintf(timer_string_aux, "%25s\t%20f\t%19.2f%%\n", "memory_transfers", timer_read(TIMER_MEMORY_TRANSFERS), (timer_read(TIMER_MEMORY_TRANSFERS)*100/timer_read(TIMER_TOTAL)));
	strcat(timer_string, timer_string_aux);
	sprintf(timer_string_aux, "%25s\t%20f\t%19.2f%%\n", "linearization", timer_read(TIMER_LINEARIZATION), (timer_read(TIMER_LINEARIZATION)*100/timer_read(TIMER_TOTAL)));
	strcat(timer_string, timer_string_aux);
	sprintf(timer_string_aux, "%25s\t%20f\t%19.2f%%\n", "deslinearization", timer_read(TIMER_DESLINEARIZATION), (timer_read(TIMER_DESLINEARIZATION)*100/timer_read(TIMER_TOTAL)));
	strcat(timer_string, timer_string_aux);
	sprintf(timer_string_aux, "%25s\t%20f\t%19.2f%%", "matrix_multiplication", timer_read(TIMER_COMPUTATION), (timer_read(TIMER_COMPUTATION)*100/timer_read(TIMER_TOTAL)));
	strcat(timer_string, timer_string_aux);
}

void debug_results(int** matrix){
	if(debug_flag){
		FILE* file;
		file = fopen("matrix_multiplication.debug.dat", "w");
		fprintf(file, "%s\n\n", checksum_string);
		for(int i=0; i<N; i++){				
			for(int j=0; j<N; j++){
				fprintf(file, "%d ", matrix[i][j]);
			}
			fprintf(file, "\n");
		}
		fclose(file);	
	}			
}

void release_resources(int** matrix1, int** matrix2, int** matrix3){
	for (int i=0; i<N; i++){
		free(matrix1[i]);
		free(matrix2[i]);
		free(matrix3[i]);
	}
	free(matrix1);
	free(matrix2);
	free(matrix3);
}
