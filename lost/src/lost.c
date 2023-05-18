/*******************************************************************************
 *      Lost -- A fast toolkit for Log-Linear models
 *
 * Copyright (c) 2012-2022  LIMSI-CNRS
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <pthread.h>

#define LOST_VERSION "0.83"
#define MAX_REAL 0

/*******************************************************************************
 * Toolbox
 ******************************************************************************/

#ifndef ENOMEM
  #define ENOMEM 0xBEEF
#endif
#define EZEPFMT (0xBEEF + 1)
#define EZEPFST (0xBEEF + 2)

#define EPSILON (DBL_EPSILON * 64.0)

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

/* fatal:
 *   This is the main error function, it will print the given message with same
 *   formating than the printf family and exit program with an error. We let the
 *   OS care about freeing ressources.
 */
static
void fatal(const char *msg, ...) {
	assert(msg != NULL);
	va_list args;
	fprintf(stderr, "error: ");
	va_start(args, msg);
	vfprintf(stderr, msg, args);
	va_end(args);
	fprintf(stderr, "\n");
	exit(EXIT_FAILURE);
}

/* pfatal:
 *   This one is very similar to the fatal function but print an additional
 *   system error message depending on the errno. This can be used when a
 *   function who set the errno fail to print more detailed informations. You
 *   must be carefull to not call other function that might reset it before
 *   calling pfatal.
 */
static
void pfatal(const char *msg, ...) {
	assert(msg != NULL);
	const char *err = NULL;
	switch (errno) {
		case ENOMEM:  err = "out of memory"; break;
		case EZEPFMT: err = "format error";  break;
		default:      err = strerror(errno);
	}
	va_list args;
	fprintf(stderr, "error: ");
	va_start(args, msg);
	vfprintf(stderr, msg, args);
	va_end(args);
	fprintf(stderr, "\n\t<%s>\n", err);
	exit(EXIT_FAILURE);
}

/*******************************************************************************
 * Threads
 ******************************************************************************/

/* atm_*
 *   Atomic primitives. This is architecture dependent code that must be adapted
 *   if you want this code to work properly. You must have hardware support for
 *   them or the code below will be extremely slow and using a lock based hash
 *   table will be a better choice in this case.
 */
#define atm_cas __sync_bool_compare_and_swap
#define atm_add __sync_add_and_fetch
#define atm_sub __sync_sub_and_fetch
#define atm_syn __sync_synchronize

static inline
void atm_inc(volatile double *value, double inc) {
	while (1) {
		volatile union {
			double   d;
			uint64_t u;
		} old, new;
		old.d = *value;
		new.d = old.d + inc;
		uint64_t *ptr = (uint64_t *)value;
		if (atm_cas(ptr, old.u, new.u))
			break;
	}
}

/* Threads, mutexes, and conditions:
 *   Simple macro wrapper around the POSIX threads interface. This simplify the
 *   usage of threads and also allow to port Lost more easily to other threading
 *   library as you just need to redefine these macros.
 */
#define thread_t pthread_t
#define thread_spawn(t, f, d) do {                                   \
	pthread_attr_t attr;                                         \
	if (pthread_attr_init(&attr) != 0)                           \
		fatal("failed to create thread attributes");         \
	pthread_attr_setstacksize(&attr, 50 * 1024 * 1024);          \
	pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);          \
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); \
	if (pthread_create((t), &attr, (f), (d)))                    \
		fatal("failed to create thread");                    \
	if (pthread_attr_destroy(&attr) != 0)                        \
		fatal("failed to destroy thread attributes");        \
} while (0)
#define thread_join(t) do {                                          \
	if (pthread_join(t, NULL))                                   \
		fatal("failed to join thread");                      \
} while (0);

#define mtx_t pthread_mutex_t
#define mtx_init(m)   do {                        \
	if (pthread_mutex_init((m), NULL))        \
		fatal("failed to create mutex");  \
} while (0)
#define mtx_clear(m)  do {                        \
	if (pthread_mutex_destroy((m)))           \
		fatal("failed to destroy mutex"); \
} while (0)
#define mtx_lock(m)   do {                        \
	if (pthread_mutex_lock((m)))              \
		fatal("failed to lock mutex");    \
} while (0)
#define mtx_unlock(m) do {                        \
	if (pthread_mutex_unlock((m)))            \
		fatal("failed to unlock mutex");  \
} while (0)

#define cond_t pthread_cond_t
#define cond_init(c)   do {                        \
	if (pthread_cond_init((c), NULL))          \
		fatal("failed to create cond");    \
} while (0)
#define cond_clear(c)  do {                        \
	if (pthread_cond_destroy((c)))             \
		fatal("failed to destroy cond");   \
} while (0)
#define cond_wait(c, m) do {                       \
	if (pthread_cond_wait((c), (m)))           \
		fatal("failed to wait for cond");  \
} while (0);
#define cond_signal(c) do {                        \
	if (pthread_cond_signal((c)))              \
		fatal("failed to signal cond");    \
} while (0);
#define cond_broadcast(c) do {                     \
	if (pthread_cond_broadcast((c)))           \
		fatal("failed to broadcast cond"); \
} while (0);

/*******************************************************************************
 * Spooky hash
 *
 *   This is the spooky hash function from Bob Jenkins, it is a very strong and
 *   fast hash suitable for use with power of two sized hash tables. The two
 *   convenience functions for buffers and strings take care of masking out the
 *   high order bit which is reserved for the hash table implementation.
 *
 *   WARNING: This code rely on unaligned reads and little endian architecture
 *   so it will need to be adapted if one these assumptions are false.
 ******************************************************************************/

typedef uint64_t hsh_t;

/* hsh_spooky:
 *   This is the Spooky hash function of Bob Jenkis implemented with low startup
 *   overhead so give good performance for small keys. The implementation mostly
 *   come from the example code v2.
 */
static
uint64_t hsh_spooky(const void *buf, const size_t len) {
	assert(buf != NULL);
	union {
		const uint8_t  *p8;
		const uint16_t *p16;
		const uint32_t *p32;
		const uint64_t *p64;
	} key = {.p8 = buf};
	const uint64_t foo = 0xDEADBEEFCAFEBABEULL;
	uint64_t tlen = len;
	uint64_t a = foo, b = foo;
	uint64_t c = foo, d = foo;
	while (tlen >= 32) {
		c += key.p64[0]; d += key.p64[1];
		c = (c << 50) | (c >> (64 - 50)); c += d; a ^= c;
		d = (d << 52) | (d >> (64 - 52)); d += a; b ^= d;
		a = (a << 30) | (a >> (64 - 30)); a += b; c ^= a;
		b = (b << 41) | (b >> (64 - 41)); b += c; d ^= b;
		c = (c << 54) | (c >> (64 - 54)); c += d; a ^= c;
		d = (d << 48) | (d >> (64 - 48)); d += a; b ^= d;
		a = (a << 38) | (a >> (64 - 38)); a += b; c ^= a;
		b = (b << 37) | (b >> (64 - 37)); b += c; d ^= b;
		c = (c << 62) | (c >> (64 - 62)); c += d; a ^= c;
		d = (d << 34) | (d >> (64 - 34)); d += a; b ^= d;
		a = (a <<  5) | (a >> (64 -  5)); a += b; c ^= a;
		b = (b << 36) | (b >> (64 - 36)); b += c; d ^= b;
		a += key.p64[2]; b += key.p64[3];
		tlen -= 32; key.p64 += 4;
	}
	if (tlen >= 16) {
		c += key.p64[0], d += key.p64[1];
		c = (c << 50) | (c >> (64 - 50)); c += d; a ^= c;
		d = (d << 52) | (d >> (64 - 52)); d += a; b ^= d;
		a = (a << 30) | (a >> (64 - 30)); a += b; c ^= a;
		b = (b << 41) | (b >> (64 - 41)); b += c; d ^= b;
		c = (c << 54) | (c >> (64 - 54)); c += d; a ^= c;
		d = (d << 48) | (d >> (64 - 48)); d += a; b ^= d;
		a = (a << 38) | (a >> (64 - 38)); a += b; c ^= a;
		b = (b << 37) | (b >> (64 - 37)); b += c; d ^= b;
		c = (c << 62) | (c >> (64 - 62)); c += d; a ^= c;
		d = (d << 34) | (d >> (64 - 34)); d += a; b ^= d;
		a = (a <<  5) | (a >> (64 -  5)); a += b; c ^= a;
		b = (b << 36) | (b >> (64 - 36)); b += c; d ^= b;
		tlen -= 16; key.p64 += 2;
	}
	d += (const uint64_t)tlen << 56;
	switch (tlen) {
		case 15: d += (const uint64_t)key.p8[14] << 48;
		case 14: d += (const uint64_t)key.p8[13] << 40;
		case 13: d += (const uint64_t)key.p8[12] << 32;
		case 12: d += key.p32[2]; c += key.p64[0];
			break;
		case 11: d += (const uint64_t)key.p8[10] << 16;
		case 10: d += (const uint64_t)key.p8[ 9] <<  8;
		case  9: d += (const uint64_t)key.p8[ 8];
		case  8: c += key.p64[0];
			break;
		case  7: c += (const uint64_t)key.p8[ 6] << 48;
		case  6: c += (const uint64_t)key.p8[ 5] << 40;
		case  5: c += (const uint64_t)key.p8[ 4] << 32;
		case  4: c += key.p32[0];
			break;
		case  3: c += (const uint64_t)key.p8[ 2] << 16;
		case  2: c += (const uint64_t)key.p8[ 1] <<  8;
		case  1: c += (const uint64_t)key.p8[ 0];
			break;
		case  0: c += foo; d += foo;
	}
	d ^= c; c = (c << 15) | (c >> (64 - 15)); d += c;
	a ^= d; d = (d << 52) | (d >> (64 - 52)); a += d;
	b ^= a; a = (a << 26) | (a >> (64 - 26)); b += a;
	c ^= b; b = (b << 51) | (b >> (64 - 51)); c += b;
	d ^= c; c = (c << 28) | (c >> (64 - 28)); d += c;
	a ^= d; d = (d <<  9) | (d >> (64 -  9)); a += d;
	b ^= a; a = (a << 47) | (a >> (64 - 47)); b += a;
	c ^= b; b = (b << 54) | (b >> (64 - 54)); c += b;
	d ^= c; c = (c << 32) | (c >> (64 - 32)); d += c;
	a ^= d; d = (d << 25) | (d >> (64 - 25)); a += d;
	b ^= a; a = (a << 63) | (a >> (64 - 63)); b += a;
	return a;
}

/* hsh_buffer:
 *   Compute the hash of a raw buffer of given size and take care of masking the
 *   63 low bits.
 */
static
hsh_t hsh_buffer(const void *buf, const size_t size) {
	assert(buf != NULL);
	const uint64_t h = hsh_spooky(buf, size);
	return h & UINT64_C(0x7FFFFFFFFFFFFFFF);
}

/* hsh_string:
 *   Compute the hash of a NUL-terminated string and take care of masking the 63
 *   low bits.
 */
static
hsh_t hsh_string(const char *str) {
	assert(str != NULL);
	const size_t   s = strlen(str);
	const uint64_t h = hsh_spooky(str, s);
	return h & UINT64_C(0x7FFFFFFFFFFFFFFF);
}

/*******************************************************************************
 * Optimized bit operations
 *
 *   The lock-free linked-list and hash-table implementation require some bit
 *   twidling implemented here. These operations are quite critical for good
 *   performances so optimized assembly versions are provided as the C only
 *   versions may be quite slow.
 ******************************************************************************/

/* bit_reverse:
 *   Reverse the order of the bits in the value [v] with some efficient hack. We
 *   first swap bit in packet of size 2, next with swap these packet inside
 *   bigger packet of size 4 and so one.
 */
static
uint64_t bit_reverse(uint64_t v) {
#ifndef __x86_64
	static const uint64_t magic[] = {
		UINT64_C(0x5555555555555555), UINT64_C(0x3333333333333333),
		UINT64_C(0x0F0F0F0F0F0F0F0F), UINT64_C(0x00FF00FF00FF00FF),
		UINT64_C(0x0000FFFF0000FFFF),
	};
	v = ((v >>  1) & magic[0]) | ((v & magic[0]) <<  1); // BS1
	v = ((v >>  2) & magic[1]) | ((v & magic[1]) <<  2); // BS2
	v = ((v >>  4) & magic[2]) | ((v & magic[2]) <<  4); // BS4
	v = ((v >>  8) & magic[3]) | ((v & magic[3]) <<  8); // BS8
	v = ((v >> 16) & magic[4]) | ((v & magic[4]) << 16); // BS16
	v = ((v >> 32)           ) | ((v           ) << 32); // BS32
	return v;
#else
	uint64_t r;
	__asm__ (
		// Swap 1-bit pairs, take input in rdi and store swapped result
		// in rdx.
		"	movabsq  $0x5555555555555555, %%rcx \n"
		"	movabsq  $0xAAAAAAAAAAAAAAAA, %%rdx \n"
		"	andq     %%rdi, %%rcx \n"
		"	andq     %%rdi, %%rdx \n"
		"	shlq     $1,    %%rcx \n"
		"	shrq     $1,    %%rdx \n"
		"	orq      %%rcx, %%rdx \n"
		// Swap 2-bit pairs, take input in rdx and store swapped result
		// in rcx.
		"	movabsq  $0x3333333333333333, %%rdi \n"
		"	movabsq  $0xCCCCCCCCCCCCCCCC, %%rcx \n"
		"	andq     %%rdx, %%rdi \n"
		"	andq     %%rdx, %%rcx \n"
		"	shlq     $2,    %%rdi \n"
		"	shrq     $2,    %%rcx \n"
		"	orq      %%rdi, %%rcx \n"
		// Swap 4-bit pairs, take input in rcx and store swapped result
		// in rdi.
		"	movabsq  $0x0F0F0F0F0F0F0F0F, %%rdx \n"
		"	movabsq  $0xF0F0F0F0F0F0F0F0, %%rdi \n"
		"	andq     %%rcx, %%rdx \n"
		"	andq     %%rcx, %%rdi \n"
		"	shlq     $4,    %%rdx \n"
		"	shrq     $4,    %%rdi \n"
		"	orq      %%rdx, %%rdi \n"
		// And finally reverse byte order with a single instruction to
		// produce the final result.
		"	bswapq   %%rdi        \n"
		: "=D"(r)
		: "D"(v)
		: "rcx", "rdx"
	);
	return r;
#endif
}

/* bit_clearmsb:
 *   Clear the most signficant bit set in the value [v]. This is done by
 *   building a mask of 1 from the most significant bit set to the lowest bit
 *   and anding this mask shifted by one with the value effectively discarding
 *   only the most significant one.
 */
static
uint64_t bit_clearmsb(uint64_t v) {
#ifndef __x86_64
	uint64_t t = v;
	t |= t >>  1; t |= t >>  2;
	t |= t >>  4; t |= t >>  8;
	t |= t >> 16; t |= t >> 32;
	return v & (t >> 1);
#else
	uint64_t r;
	__asm__ (
		"	bsrq  %%rdi, %%rcx \n"
		"	movq  $1,    %%rax \n"
		"	shlq  %%cl,  %%rax \n"
		"	xorq  %%rdi, %%rax \n"
		: "=a"(r)
		: "D"(v)
		: "rcx"
	);
	return r;
#endif
}

/*******************************************************************************
 * Lock-free sorted linked list
 *   HC SVNT DRACONES
 *
 *   This is an implementation of a lock-free sorted singly linked-list based on
 *   the work of [1]. This implementation allow searching, inserting, and
 *   deleting nodes concurently without locking.
 *   This implementation require each list to start with a dummy head node whose
 *   key is ignored. This simplify and speed-up the code by removing the
 *   handling of corner cases and doesn't impact the code as thoses nodes are
 *   implicitly added by the hash table.
 *
 *  [1] High Performance Dynamic Lock-Free Hash Tables and List-Based Sets,
 *      Maged M. Michael, ACM Symposium on Parallelism in Algorithms and
 *      Architectures, august 2002.
 ******************************************************************************/

/* lst_t:
 *   Working data that should be embeded at start of objects that need to be
 *   inserted in lock-free lists. The fields should not be changed as the
 *   assembly code rely on them being exactly as is.
 */
typedef struct lst_s lst_t;
struct lst_s {
	lst_t *next;  // Pointer to the next item
	hsh_t  key;   // Key of the current item
};

/* ptr_*:
 *   These three macros are used to handle tagged pointers. They use the lowest
 *   bit of the pointer to store the tag and so assume that this bit will always
 *   be zero as a result of alignment. If this is not valid on your architecture
 *   but other bit are free to use, you have to adjust these.
 */
#define ptr_addtag(n) ((void *)((uintptr_t)(n) |  1))
#define ptr_remtag(n) ((void *)((uintptr_t)(n) & ~1))
#define ptr_tagged(n) ((int   )((uintptr_t)(n) &  1))

/* lst_search:
 *   Search for a node with the given key in the list. In all cases it also
 *   return an array of three nodes with the guarantee that, at some point
 *   during the call, the following conditions were satisfied concurently:
 *     - the three nodes form a sequence of consecutive nodes ;
 *     - the first two nodes are not marked for deletion ;
 *     - the key of second node is the searched key if true is returned or the
 *       lowest key greater than it if false is returned.
 */
static
int lst_search(lst_t *head, const hsh_t key, lst_t *ptr[3]) {
	assert(head != NULL && ptr != NULL);
#ifndef __x86_64
	do {
		// This outer loop setup the 0->1 line at the start of the list
		// and enter the inner loop who build the full chain and check
		// the links. Each time the search must start again due to a
		// broken link we go back here.
		ptr[0] = head;
		ptr[1] = head->next;
		do {
			ptr[1] = ptr_remtag(ptr[1]);
			if (ptr[1] == NULL)
				return 0;
			// Setup the 1->2 link and check that the 0->1 link is
			// still valid. If the check fail we have to start again
			// and hope for more luck, else we know that when the
			// key was retrieved the chain was valid.
			const hsh_t ckey = ptr[1]->key; atm_syn();
			ptr[2] = ptr[1]->next;
			if (ptr[0]->next != ptr[1])
				break;
			// If node 2 is not marked for deletion, we check if we
			// have found the one we search or if we have gone too
			// far in the list. In both case we return to the user,
			// else we just adavance one step in the list.
			if (!ptr_tagged(ptr[2])) {
				if (ckey >= key)
					return ckey == key;
				ptr[0] = ptr[1];
				ptr[1] = ptr[2];
				continue;
			}
			// Else, if the node is tagged we have to try to remove
			// it. If we fail, the chain is broken and we have to
			// start again, else we just fix the chain and keep
			// searching.
			ptr[2] = ptr_remtag(ptr[2]);
			if (!atm_cas(&(ptr[0])->next, ptr[1], ptr[2]))
				break;
			ptr[1] = ptr[2];
		} while (1);
	} while (1);
#else
	int res;
	__asm__ (
		// This code reserve the three register (r8, r9, r10) to store
		// the ptr[3] array while looping. Other registers are used as
		// described by the constrains section, and (rcx) is used to
		// store the key value around the sync point.
		"	xorq     %%rax,    %%rax   \n"
		// Special loop entry point: setup the 0->1 link at the head of
		// the loop and jump over loop header.
		"1:	movq     %%rdi,    %%r8    \n"
		"	movq    (%%rdi),   %%r9    \n"
		"	jmp      5f                \n"
		// If we come here, the r9 node is marked for deletion, so we
		// try to remove it from the list
		"7:	andq     $-2,      %%r10   \n"
		"	movq     %%r9,     %%rax   \n"
		"	lock                       \n"
		"	cmpxchgq %%r10,   (%%r8)   \n"
		"	cmpq     %%r9,     %%rax   \n"
		"	jne      1b                \n"
		"	jmp      4f                \n"
		// Loop header:
		//   this code advance one step in the list and build a valid
		//   chain of nodes. There is three entry point in this header
		//   for (next), (skip) and (none) moves.
		"	.align 4, 90               \n"
		"3:	movq     %%r9,     %%r8    \n" // ptr[0] = ptr[1]
		"4:	movq     %%r10,    %%r9    \n" // ptr[1] = ptr[2]
		"5:	andq     $-2,      %%r9    \n"
		"	jz       9f                \n"
		"	movq   8(%%r9),    %%rcx   \n" // ckey   = ptr[1]->key
		"	movq    (%%r9),    %%r10   \n" // ptr[2] = ptr[1]->next
		"	cmpq    (%%r8),    %%r9    \n"
		"	jne      1b                \n"
		// Loop check:
		//   if ptr[1] is marked for removal jump to handler ;
		//   if ptr[1]->key still to low jump to next item ;
		//   else exit loop.
		"	testb    $1,       %%r10b  \n"
		"	jne      7b                \n"
		"	cmpq     %%rsi,    %%rcx   \n"
		"	jb       3b                \n"
		// Exit point:
		//   The result is stored in al and the pointer chain is moved
		//   to the caller supplyed array.
		"	sete     %%al              \n"
		"9:	movq     %%r8,    (%%rdx)  \n"
		"	movq     %%r9,   8(%%rdx)  \n"
		"	movq     %%r10, 16(%%rdx)  \n"
		: "=a"(res)
		: "D"(head), "S"(key), "d"(ptr)
		: "rcx", "r8", "r9", "r10", "memory"
	);
	return res;
#endif
}

/* lst_find:
 *   Find in the list a node with the given key. Return true and a reference to
 *   it if the node is found, and false if not. If the reference is not needed
 *   [res] may be NULL.
 */
static
int lst_find(lst_t *head, const hsh_t key, lst_t **res) {
	assert(head != NULL);
	lst_t *ptr[3];
	if (lst_search(head, key, ptr)) {
		if (res != NULL)
			*res = ptr[1];
		return 1;
	}
	if (res != NULL)
		*res = NULL;
	return 0;
}

/* lst_insert:
 *   Insert in the list the given node in ordered position relative to its key
 *   field. If its key is not already present, the node is inserted and true is
 *   returned, else a pointer to the node with the same key is stored in [res]
 *   and false is returned.
 */
static
int lst_insert(lst_t *head, lst_t *node, lst_t **res) {
	assert(head != NULL && node != NULL);
	// Repeat until success. As other thread may work concurently on the
	// table we may fail to insert and have to retry it.
	do {
		lst_t *ptr[3];
		// First check if the key is already present in the list. In
		// this case the insertion fail and return to the caller.
		if (lst_search(head, node->key, ptr)) {
			if (res != NULL)
				*res = ptr[1];
			return 0;
		}
		// Else, try to insert the new key at its position, if this fail
		// the chain build by the search function was broken and so we
		// have to try again.
		node->next = ptr[1];
		if (atm_cas(&(ptr[0])->next, ptr[1], node)) {
			if (res != NULL)
				*res = node;
			return 1;
		}
	} while (1);
}

/* lst_remove:
 *   Remove the node with given key from the list. If the node is found, a
 *   reference to it is stored in [res], the node is unlinked from the list and
 *   true is returned, else false is returned.
 *   On success, the caller is responsible to keep the node available until all
 *   operations on the list started before the end of the call are finished as
 *   they may retain reference to this node.
 */
static
int lst_remove(lst_t *head, const hsh_t key, lst_t **res) {
	assert(head != NULL);
	lst_t *ptr[3], *mark = NULL;
	// First, search the node to be deleted and mark it for deletion. We
	// retry until we succeed or until the node is not found in the list.
	do {
		if (!lst_search(head, key, ptr)) {
			if (res != NULL)
				*res = NULL;
			return 0;
		}
		mark = ptr_addtag(ptr[2]);
	} while (!atm_cas(&(ptr[1])->next, ptr[2], mark));
	// Now the node is marked, try to remove it from the list. If the cas
	// fail we search for the node as the search function will take care of
	// ensuring the node is trully removed from the list.
	if (!atm_cas(&(ptr[0])->next, ptr[1], ptr[2]))
		lst_search(head, key, ptr);
	if (res != NULL)
		*res = ptr[1];
	return 1;
}

/*******************************************************************************
 * Lock-free hash table
 *
 *   This is an implementation of a lock-free hash table based on the work
 *   described in [1]. This implementation allow lookup, insertion, and deletion
 *   of key, values pairs concurently without locking.
 *   Deletion is restricted for now as the node is safely removed from the table
 *   but its memory can only be reclaimed when all operations started before the
 *   removal are finished.
 *
 *   The key is assumed to be a 63 bit hash with uniformly distributed values
 *   suitable for a masking scheme on the low-order bits. This mean that the
 *   highest bit of the 64bits value is reserved for internal use and not
 *   preserved, and if some bits are less random they should be in the high part
 *   of the value. As the table grow based on the mean list length, if the hash
 *   function is not uniform bad performance should be expected.
 *
 *  [1] Split-Ordered Lists: Lock-Free Extensible Hash Tables, Ori Shalev and
 *      Nir Shavit, May 2006. Journal of the ACM, Vol. 53, No. 3, pp. 379–405.
 ******************************************************************************/

/* map_t:
 *   Lock-free hash table object storing the main list, the bucket table, and
 *   various information datas.
 */
typedef struct map_s map_t;
struct map_s {
	lst_t    list;    // Head of the full item list
	lst_t ***bucket;  // Two-level indirect bucket heads table
	size_t   size;    // Current size of the bucket table
	size_t   count;   // Number of items currently stored in the table
	size_t   grow;    // Maximum mean list length before growing
};

/* key_*:
 *   Various macros to convert from hash values given by the user to key in
 *   reverse order for the list and tagged values for head nodes.
 */
#define key_normal(k) (bit_reverse((k)) |  1)
#define key_marker(k) (bit_reverse((k)) & ~1)
#define key_tohash(k) (bit_reverse((k) & ~1))
#define key_ismark(k) (!((k) & 1))

/* map_new:
 *   Allocate a new empty hash table and return it. If allocation of the table
 *   fail, NULL is returned.
 */
static
map_t *map_new(void) {
	// First allocate a new empty table object. After this point the table
	// is still invalid as the hash part is not initialized.
	map_t *map = malloc(sizeof(map_t));
	if (map == NULL) {
		errno = ENOMEM;
		return NULL;
	}
	map->list.next = NULL;
	map->list.key  = 0;
	map->bucket    = NULL;
	map->size      = 0x10;
	map->count     = 0;
	map->grow      = 8;
	// Allocate the two level bucket table. On the second level we allocate
	// the first segment as required for a valid hash table.
	lst_t ***tbl = malloc(sizeof(lst_t **) * 0x10000);
	lst_t  **seg = malloc(sizeof(lst_t  *) * 0x10000);
	if (tbl == NULL || seg == NULL) {
		free(tbl); free(seg);
		free(map);
		errno = ENOMEM;
		return NULL;
	}
	for (int i = 0; i < 0x10000; i++)
		tbl[i] = NULL, seg[i] = NULL;
	map->bucket    = tbl;
	map->bucket[0] = seg;
	// And finally, setup the first bucket of the first segment. If this
	// succeed the table is now valid as the root bucket is initialized.
	lst_t *bkt = malloc(sizeof(lst_t));
	if (bkt == NULL) {
		free(bkt); free(tbl);
		free(seg); free(map);
		errno = ENOMEM;
		return NULL;
	}
	bkt->key = key_marker(0);
	lst_insert(&map->list, bkt, NULL);
	map->bucket[0][0] = bkt;
	return map;
}

/* map_free:
 *   Free all memory used by the table. Caller must ensure that the table is not
 *   used anymore by any threads and should take care of freeing the values
 *   present in the table.
 */
static
void map_free(map_t *map, void (*dst)(void *)) {
	if (map == NULL)
		return;
	if (dst != NULL) {
		lst_t *nd = map->list.next;
		while (nd != NULL) {
			lst_t *nxt = nd->next;
			if (!key_ismark(nd->key))
				dst(nd);
			nd = nxt;
		}
	}
	for (int seg = 0; seg < 0x10000; seg++) {
		if (map->bucket[seg] == NULL)
			break;
		for (int bkt = 0; bkt < 0x10000; bkt++)
			if (map->bucket[seg][bkt] != NULL)
				free(map->bucket[seg][bkt]);
		free(map->bucket[seg]);
	}
	free(map->bucket);
	free(map);
}

/* map_getbkt:
 *   Return the list head for the bucket. This ensure that the segment and
 *   list head are initialized doing it if needed. If an allocation fail while
 *   creating a new bucket, the previous bucket is returned allowing the table
 *   to still works, just more slowly. This mean that this function could never
 *   fail.
 */
static
lst_t *map_getbkt(map_t *map, const hsh_t bkt) {
	assert(map != NULL && bkt < UINT64_C(0x100000000));
	const hsh_t seg = bkt / 0x10000;
	const hsh_t idx = bkt % 0x10000;
	// We first check if the segment containing the bucket is available or
	// try to create it if not.
	if (map->bucket[seg] == NULL) {
		lst_t **tmp = malloc(sizeof(lst_t *) * 0x10000);
		if (tmp == NULL)
			return map_getbkt(map, bit_clearmsb(bkt));
		for (int i = 0; i < 0x10000; i++)
			tmp[i] = NULL;
		if (!atm_cas(&map->bucket[seg], NULL, tmp))
			free(tmp);
	}
	// Next we check if the bucket itself is initialized and if not we have
	// to do it.
	if (map->bucket[seg][idx] == NULL) {
		lst_t *prev = map_getbkt(map, bit_clearmsb(bkt));
		lst_t *cbkt = malloc(sizeof(lst_t));
		lst_t *res  = NULL;
		if (cbkt == NULL)
			return prev;
		cbkt->key = key_marker(bkt);
		if (!lst_insert(prev, cbkt, &res))
			free(cbkt);
		map->bucket[seg][idx] = res;
	}
	// Now, all should be initialized and valid so we can return the bucket
	// to the caller.
	return map->bucket[seg][idx];
}

/* map_find:
 *   Search the table for the given key. If the key is found, a reference to its
 *   value is returned, else NULL is returned.
 */
static
void *map_find(map_t *map, const hsh_t hash) {
	assert(map != NULL);
	const hsh_t bkt = hash & (hsh_t)(map->size - 1);
	const hsh_t key = key_normal(hash);
	lst_t *head = map_getbkt(map, bkt);
	lst_t *res  = NULL;
	if (lst_find(head, key, &res))
		return res;
	return NULL;
}

/* map_insert:
 *   Insert the value with a new key in the table if this key is not already
 *   present in the table and return a reference to the value associated with
 *   the key. (either the new one or an already present one)
 */
static
void *map_insert(map_t *map, const hsh_t hash, void *val) {
	assert(map != NULL && val != NULL);
	const hsh_t bkt = hash & (hsh_t)(map->size - 1);
	const hsh_t key = key_normal(hash);
	lst_t *node = (lst_t *)val;
	lst_t *head = map_getbkt(map, bkt);
	lst_t *res  = NULL;
	node->key = key;
	if (lst_insert(head, val, &res)) {
		const size_t size = map->size;
		if (atm_add(&map->count, (size_t)1) / size > map->grow)
			atm_cas(&map->size, size, size * 2);
	}
	return res;
}

/* map_remove:
 *   Remove the value associated with the key from the table. If the key is
 *   found, a reference to the removed value is returned, else NULL is returned.
 *   WARNING: Before freeing the value returned, the caller must ensure that any
 *   operation on this table started before the return of this call have also
 *   returned.
 */
static
void *map_remove(map_t *map, const hsh_t hash) {
	assert(map != NULL);
	const hsh_t bkt = hash & (hsh_t)(map->size - 1);
	const hsh_t key = key_normal(hash);
	lst_t *head = map_getbkt(map, bkt);
	lst_t *res  = NULL;
	if (lst_remove(head, key, &res)) {
		atm_sub(&map->count, (size_t)1);
		return res;
	}
	return NULL;
}

/* map_next:
 *   Iterator over all items stored in the map. If [last] is NULL, this return
 *   the first item, else return the item following [last] if any, or NULL if it
 *   is the last one.
 */
static
void *map_next(map_t *map, void *last) {
	assert(map != NULL);
	lst_t *nd = last;
	if (nd == NULL)
		nd = &map->list;
	do {
		nd = nd->next;
	} while (nd != NULL && key_ismark(nd->key));
	return nd;
}

/* map_gethsh:
 *   Return the hash of a value returned by the iterator.
 */
static
hsh_t map_gethsh(const void *val) {
	assert(val != NULL);
	const lst_t *node = (lst_t *)val;
	return key_tohash(node->key);
}

/*******************************************************************************
 * Vocabulary
 *
 *   Implement vocabulary: mapping between strings and identifiers in both
 *   directions. The mapping between strings to identifiers is done with splay
 *   tree [1], a kind of self balancing search tree. The identifier to string
 *   mapping is done with a simple vector with pointer to the key interned in
 *   the tree node, so pointer returned are always constant.
 *
 *   This object is not fully thread-safe and is intended to store only small
 *   and non-critical databases. Only the reverse mapping from identifiers to
 *   strings can be used by multiples threads and only if no new strings are
 *   added anymore.
 *
 *   [1] Sleator, Daniel D. and Tarjan, Robert E. ; Self-adjusting binary search
 *   trees, Journal of the ACM 32 (3): pp. 652--686, 1985. DOI:10.1145/3828.3835
 ******************************************************************************/

/* vnd_t:
 *   Node of the splay tree who hold a (key, value) pair. The left and right
 *   childs are stored in an array so code for each side can be factorized.
 */
typedef struct vnd_s vnd_t;
struct vnd_s {
	vnd_t *child[2]; // Left and right childs of the node
	int    value;    // Value stored in the node
	char   key[];    // The key directly stored in the node
};

/* voc_t:
 *   The vocabulary with his [tree] and [vector]. The database hold [count]
 *   (key, value) pairs, but the vector is of size [size], it will grow as
 *   needed.
 */
typedef struct voc_s voc_t;
struct voc_s {
	vnd_t  *tree;    // The tree for direct mapping
	vnd_t **vect;    // The array for the reverse mapping
	int     count;   // The number of items in the database
	int     size;    // The real size of [vector]
};

/* voc_new:
 *   Create a new empty vocabulary ready for doing mappings.
 */
static
voc_t *voc_new(void) {
	voc_t *voc = malloc(sizeof(voc_t));
	if (voc == NULL) {
		errno = ENOMEM;
		return NULL;
	}
	voc->tree  = NULL;
	voc->vect  = NULL;
	voc->count = 0;
	voc->size  = 0;
	return voc;
}

/* voc_free:
 *   Free a vocabulary and all associated memory. All return strings become
 *   invalid after this call.
 */
static
void voc_free(voc_t *voc) {
	assert(voc != NULL);
	if (voc->tree != NULL) {
		assert(voc->vect != NULL);
		for (int i = 0; i < voc->count; i++)
			free(voc->vect[i]);
		free(voc->vect);
	}
	free(voc);
}

/* voc_splay:
 *   Do a splay operation on the tree part of the vocab with the given key.
 *   Return -1 if the key is in the vocab, so if it has been move at the top,
 *   else return the side wether the key should go.
 */
static
int voc_splay(voc_t *voc, const char *key) {
	assert(voc != NULL && key != NULL);
	vnd_t  nil = {{NULL, NULL}, 0};
	vnd_t *root[2] = {&nil, &nil};
	vnd_t *nd = voc->tree;
	int side;
	while (1) {
		side = strcmp(key, nd->key);
		side = (side == 0) ? -1 : (side < 0) ? 0 : 1;
		if (side == -1 || nd->child[side] == NULL)
			break;
		const int tst = (side == 0)
			? strcmp(key, nd->child[side]->key) < 0
			: strcmp(key, nd->child[side]->key) > 0;
		if (tst) {
			vnd_t *tmp = nd->child[side];
			nd->child[side] = tmp->child[1 - side];
			tmp->child[1 - side] = nd;
			nd = tmp;
			if (nd->child[side] == NULL)
				break;
		}
		root[1 - side]->child[side] = nd;
		root[1 - side] = nd;
		nd = nd->child[side];
	}
	root[0]->child[1] = nd->child[0];
	root[1]->child[0] = nd->child[1];
	nd->child[0] = nil.child[1];
	nd->child[1] = nil.child[0];
	voc->tree = nd;
	return side;
}

/* voc_id2str:
 *   Return the key associated with the given identifier. The key must not be
 *   modified nor freed by the caller and remain valid for the lifetime of the
 *   vocab object. Return NULL if the identifier is invalid.
 */
/*static
const char *voc_id2str(const voc_t *voc, int id) {
	assert(voc != NULL);
	if (id < 0 || id >= voc->count)
		return NULL;
	return voc->vect[id]->key;
}*/

/* voc_newnode:
 *   Create a new node object with given key, value and no childs. The key
 *   is interned in the node so no reference is kept to the given string.
 */
static
vnd_t *voc_newnode(const char *key, int val) {
	assert(key != NULL && val >= 0);
	const int len = strlen(key) + 1;
	vnd_t *nd = malloc(sizeof(vnd_t) + len);
	if (nd == NULL) {
		errno = ENOMEM;
		return NULL;
	}
	memcpy(nd->key, key, len);
	nd->value = val;
	nd->child[0] = NULL;
	nd->child[1] = NULL;
	return nd;
}

/* voc_str2id:
 *   Return the identifier corresponding to the given key in the vocab object
 *   inserting it if needed. On error, return -1 with errno set appropriately.
 */
static
int voc_str2id(voc_t *voc, const char *key) {
	assert(voc != NULL && key != NULL);
	// if tree is empty, directly add a root. This is where the true
	// initialization of the trie is done and first allocation of the
	// inverse vector.
	if (voc->count == 0) {
		if (voc->size == 0) {
			const int size = 128;
			voc->vect = malloc(sizeof(vnd_t *) * size);
			if (voc->vect == NULL) {
				errno = ENOMEM;
				return -1;
			}
			voc->size = size;
		}
		vnd_t *nd = voc_newnode(key, 0);
		if (nd == NULL)
			return -1;
		voc->tree    = nd;
		voc->count   = 1;
		voc->vect[0] = nd;
		return 0;
	}
	// Else if key is already there, return his value. This is the easy case
	// where no failure is possible.
	const int side = voc_splay(voc, key);
	if (side == -1)
		return voc->tree->value;
	// Else, add the key to the vocab. Here we can fail while growing the
	// vector or at node allocation so must be carefull.
	if (voc->count == voc->size) {
		const int size = voc->size * 2;
		vnd_t **tmp = realloc(voc->vect, sizeof(vnd_t *) * size);
		if (tmp == NULL) {
			errno = ENOMEM;
			return -1;
		}
		voc->size = size;
		voc->vect = tmp;
	}
	int id = voc->count;
	vnd_t *nd = voc_newnode(key, id);
	if (nd == NULL)
		return -1;
	nd->child[    side] = voc->tree->child[side];
	nd->child[1 - side] = voc->tree;
	voc->tree->child[side] = NULL;
	voc->tree = nd;
	voc->vect[id] = nd;
	voc->count++;
	return id;
}

/*******************************************************************************
 * Simple string toolbox
 ******************************************************************************/

/* str_splitsp:
 *   Split a string in space separated tokens. At most [n] tokens will be put in
 *   the array. Return the number of tokens found. This modify the string itself
 *   by adding NUL char to terminate the tokens.
 */
static
int str_splitsp(char *str, int n, char *tok[n]) {
	assert(str != NULL && n > 0 && tok != NULL);
	int cnt = 0;
	while (*str != '\0' && cnt < n) {
		while (isspace(*str))
			str++;
		if (*str == '\0')
			break;
		tok[cnt++] = str;
		while (*str != '\0' && !isspace(*str))
			str++;
		if (*str == '\0')
			break;
		*str++ = '\0';
	}
	return cnt;
}

/* str_readln:
 *   Read an input line from file. The line can be of any size limited only by
 *   available memory, a buffer large enough is allocated and returned. The
 *   caller is responsible to free it. On end-of-file or error, NULL is returned
 *   and the caller must check feof or errno to differenciate.
 */
static
char *str_readln(FILE *file) {
	assert(file != NULL);
	if (feof(file))
		return NULL;
	// Initialize the buffer
	size_t len = 0, size = 16;
	char *buffer = malloc(size);
	if (buffer == NULL)
		return NULL;
	// We read the line chunk by chunk until end of line, file or error
	while (!feof(file)) {
		if (fgets(buffer + len, size - len, file) == NULL) {
			// On NULL return there is two possible cases, either an
			// error or the end of file
			if (ferror(file)) {
				free(buffer);
				return NULL;
			}
			// On end of file, we must check if we have already read
			// some data or not
			if (len == 0) {
				free(buffer);
				errno = 0;
				return NULL;
			}
			break;
		}
		// Check for end of line, if this is not the case enlarge the
		// buffer and go read more data
		len += strlen(buffer + len);
		if (len == size - 1 && buffer[len - 1] != '\n') {
			size = size * 1.4;
			char *tmp = realloc(buffer, size);
			if (tmp == NULL) {
				free(buffer);
				errno = ENOMEM;
				return NULL;
			}
			buffer = tmp;
			continue;
		}
		break;
	}
	// At this point empty line should have already catched so we just
	// remove the end of line if present and return.
	if (buffer[len - 1] == '\n')
		buffer[--len] = '\0';
	return buffer;
}

/* str_readeos:
 *   Read a set of lines finished by the EOS mark. Return it as an array of
 *   string terminated by NULL. Return NULL on end-of-file or error. Caller must
 *   check feof or errno to distinguish.
 */
static
char **str_readeos(FILE *file) {
	assert(file != NULL);
	int size = 4, cnt = 0;
	char **lines = malloc(sizeof(char *) * size);
	if (lines == NULL) {
		errno = ENOMEM;
		return NULL;
	}
	while (!feof(file)) {
		char *raw = str_readln(file);
		if (raw == NULL) {
			if (!feof(file))
				goto error;
			break;
		}
		// The check for end of sample is not simple as we want to allow
		// spaces before and after the mark. If we find the mark, we
		// just break the loop.
		char *line = raw;
		while (isspace(*line))
			line++;
		if (!strncmp("EOS", line, 3)) {
			line += 3;
			while (isspace(*line))
				line++;
			if (*line == '\0') {
				free(raw);
				break;
			}
		}
		// If needed we resize the array. We should ensure there is
		// always at least one extra slot for the final NULL.
		if (cnt == size - 1) {
			size = size * 2;
			char **tmp = realloc(lines, sizeof(char *) * size);
			if (tmp == NULL) {
				free(raw);
				errno = ENOMEM;
				goto error;
			}
			lines = tmp;
		}
		lines[cnt++] = raw;
	}
	// If no lines was read, just cleanup and return NULL. Else, add the
	// terminal NULL and return the set of lines.
	if (cnt == 0) {
		free(lines);
		return NULL;
	}
	lines[cnt] = NULL;
	return lines;
    error:
	for (int i = 0; i < cnt; i++)
		free(lines[i]);
	free(lines);
	return NULL;
}

/*******************************************************************************
 * Progress repport system
 *
 *   This is a small set of tools to display progression of long tasks. All part
 *   of lost that may take quite a long time must use this to display some
 *   activity to the user, and inform it about the progression.
 *
 *   As we generally don't known in advance the number of step needed to finish
 *   each task, we don't use scalled progress bar, only unbounded ones. User is
 *   responsible to interpret them.
 ******************************************************************************/
typedef struct prg_s prg_t;
struct prg_s {
	long   step, max;
	long   count;
	time_t start;
	time_t last;
};

/* prg_new:
 *   Allocate a new progress object with a step size of [step].
 */
static
prg_t *prg_new(long step) {
	assert(step != 0);
	prg_t *prg = malloc(sizeof(prg_t));
	prg->step = step;
	return prg;
}

/* prg_free:
 *   Free a progress object.
 */
static
void prg_free(prg_t *prg) {
	assert(prg != NULL);
	free(prg);
}

/* prg_start:
 *   Start a new progress sequence which will report progress every [step] steps
 *   of progression. Each 10 mark a separator is also output and every 50 a new
 *   lines is started.
 */
static
void prg_start(prg_t *prg) {
	assert(prg != NULL);
	fprintf(stderr, "        [");
	prg->count = 0;
	prg->start = time(NULL);
	prg->last  = time(NULL);
}

/* prg_next:
 *   Inform the progress system that a new item have been processed. This will
 *   output progress information if needed. Don't do output during a progress as
 *   the cursor will probably be somewhere on the line.
 */
static
void prg_next(prg_t *prg) {
	assert(prg != NULL);
	const long n = atm_add(&prg->count, (long)1);
	if (n % prg->step != 0)
		return;
	if (n % (50 * prg->step) == 0) {
		const time_t now = time(NULL);
		const int dlt = difftime(now, prg->last);
		const int dlts = dlt % 60, dltm = dlt / 60;
		fprintf(stderr, "-]  tm=%dm%02ds\n        [", dltm, dlts);
		prg->last = now;
		return;
	} else if (n % (10 * prg->step) == 0) {
		fprintf(stderr, "|");
	} else {
		fprintf(stderr, "-");
	}
}

/* prg_end:
 *   End a progress sequence, finish the progress line and display the total
 *   elapsed time. Cursor will be at start of next line after this call.
 */
static
void prg_end(prg_t *prg) {
	assert(prg != NULL);
	const int dlt = difftime(time(NULL), prg->start);
	const int dlts = dlt % 60, dltm = dlt / 60;
	fprintf(stderr, "]  total=%dm%02ds\n", dltm, dlts);
}

/*******************************************************************************
 * Shared string pool
 *
 *   For efficiency, the core never directly works with string but always hash
 *   values. On input, all strings are hashed and only these values are next
 *   used allowing fast comparisons and combinations.
 *   The inverse process of mapping back hash to string is almost never needed
 *   except to output a decoded FST or to dump a model. The shared string pool
 *   is responsible for storing this inverse mapping.
 *
 *   There is two kind of strings stored here, the mandatory ones for which the
 *   inverse mapping is always stored, and the optional ones for which it is
 *   stored only in dump mode.
 ******************************************************************************/

/* ssp_t:
 *   Shared string pool object. Just store the map associating the hash values
 *   to strings and a flag indicating if the optional strings should also be
 *   stored.
 */
typedef struct ssp_s ssp_t;
struct ssp_s {
	map_t *map;
	int    all;
};

/* ist_t:
 *   Simple object just for encapsulating a NUL-terminated string with a list
 *   item for insertion in the hash-table.
 */
typedef struct ist_s ist_t;
struct ist_s {
	lst_t lst;
	char  str[];
};

/* ssp_new:
 *   Allocate a new empty string pool. If [all] is true, all strings will be
 *   stored, else only the mandatory ones.
 */
static
ssp_t *ssp_new(int all) {
	ssp_t *ssp = malloc(sizeof(ssp_t));
	if (ssp == NULL) {
		errno = ENOMEM;
		return NULL;
	}
	ssp->map = map_new();
	if (ssp->map == NULL) {
		free(ssp);
		return NULL;
	}
	ssp->all = all;
	return ssp;
}

/* ssp_free:
 *   Free a shared string pool and all associated memory.
 */
static
void ssp_free(ssp_t *ssp) {
	assert(ssp != NULL && ssp->map != NULL);
	map_free(ssp->map, free);
	free(ssp);
}

/* ssp_buffer:
 *   Store the string from the given buffer in the shared string pool. The [md]
 *   parameter indicate if the string is a mandatory one. Return the hash value
 *   of the string.
 *   If a memory error happen, the hash is still returned but the errno is set
 *   so operation can continue but without the inverse mapping stored. Caller
 *   have to choose if this is fatal or not.
 */
static
hsh_t ssp_buffer(ssp_t *ssp, const void *buf, size_t size, int md) {
	assert(ssp != NULL && ssp->map != NULL);
	assert(buf != NULL);
	hsh_t hsh = hsh_buffer(buf, size);
	if (md || ssp->all) {
		if (map_find(ssp->map, hsh) == NULL) {
			ist_t *str = malloc(sizeof(ist_t) + size + 1);
			if (str == NULL) {
				errno = ENOMEM;
				return hsh;
			}
			memcpy(str->str, buf, size);
			str->str[size] = '\0';
			if (map_insert(ssp->map, hsh, str) != str)
				free(str);
		}
	}
	return hsh;
}

/* ssp_string:
 *   Do the same than [ssp_buffer] but on a NUL-terminated string.
 */
static
hsh_t ssp_string(ssp_t *ssp, const char *str, int md) {
	assert(ssp != NULL && ssp->map != NULL);
	assert(str != NULL);
	return ssp_buffer(ssp, str, strlen(str), md);
}

/* ssp_get:
 *   Return the string associated with the given hash value or a special unknown
 *   string if the hash is not known.
 */
static
const char *ssp_get(ssp_t *ssp, hsh_t hsh) {
	static const char *unk = "@@UNKNOWN";
	ist_t *ist = map_find(ssp->map, hsh);
	if (ist == NULL)
		return unk;
	return ist->str;
}

/* ssp_load:
 *   Load a set of string from the given file to the shared pool. All string are
 *   added as mandatory one. The format is simple: one string per line with
 *   empty lines ignored, the first token on each lines is ignored as it will be
 *   the hash if it come from a previous save of the pool. Failure during
 *   insertion are ignored but IO error are reported.
 *   Return true if all went OK.
 */
static
int ssp_load(ssp_t *ssp, const char *fn) {
	assert(ssp != NULL && ssp->map != NULL);
	assert(fn != NULL);
	FILE *file = fopen(fn, "r");
	if (file == NULL)
		return 0;
	while (!feof(file)) {
		errno = 0;
		char *raw = str_readln(file);
		if (raw == NULL) {
			fclose(file);
			return errno == 0;
		}
		char *line = raw;
		while (*line != '\0' && !isspace(*line))
			line++;
		while (*line != '\0' && isspace(*line))
			line++;
		if (*line != '\0')
			ssp_string(ssp, line, 1);
		free(raw);
	}
	fclose(file);
	return 1;
}

/* ssp_save:
 *   Save the current set of string to the given file. File format is simple
 *   with just one string per line in no particular order. Return true on
 *   success.
 */
static
int ssp_save(ssp_t *ssp, const char *fn) {
	assert(ssp != NULL && ssp->map != NULL);
	assert(fn != NULL);
	FILE *file = fopen(fn, "w");
	if (file == NULL)
		return 0;
	ist_t *str = map_next(ssp->map, NULL);
	while (str != NULL) {
		const hsh_t hash = map_gethsh(str);
		fprintf(file, "%016"PRIx64" %s\n", hash, str->str);
		str = map_next(ssp->map, str);
	}
	fclose(file);
	return 1;
}

/*******************************************************************************
 * Model object
 ******************************************************************************/

/* lbl_t:
 *   Label object either on input or output side of the transducers. A label is
 *   just a string but for efficiency we also store each tokens as an hash value
 *   used for fast feature generation. Each token is also stored as a string so
 *   we can output a string representation of the features.
 */
typedef struct lbl_s lbl_t;
struct lbl_s {
	lst_t lst;    // List item for insertion in hash table
	hsh_t raw;    // Hash of raw unparsed string of the label
	int   cnt;    // Number of tokens in the label
	hsh_t tok[];  // List of tokens hash values
};

/* ftr_t:
 *   Feature object. This store the feature value and gradient as well as any
 *   other value that may be needed by the optimizer.
 */
typedef struct ftr_s ftr_t;
struct ftr_s {
	lst_t lst;
	double x;
	double g;
	// Stuff needed by the optimizer
	float  gp;   // Value of the gradient on previous iteration
	float  stp;  // Current step value on this dimension
	float  dlt;  // Value of the previous update that can be undone
	int    frq;  // Feature frequency
};

typedef struct mdl_s mdl_t;
struct mdl_s {
	map_t *ftrs;
	ssp_t *ssp;  // Shared string pool
	map_t *src;  // Source label vocabulary <str,lbl_t>
	map_t *trg;  // Target label vocabulary <str,lbl_t>
	ftr_t *real[MAX_REAL];
	int    itr;
	int    frq;
	int    stt[128];
	int    rem[128];
	FILE  *dump;
};

/* mdl_new:
 *   Create a new empty model ready to accept new features before being trained
 *   or used to decode new samples.
 */
static
mdl_t *mdl_new(ssp_t *ssp) {
	mdl_t *mdl = malloc(sizeof(mdl_t));
	if (mdl == NULL) {
		errno = ENOMEM;
		return NULL;
	}
	mdl->ssp  = ssp;
	mdl->src  = map_new();
	mdl->trg  = map_new();
	mdl->ftrs = map_new();
	if (mdl->ftrs == NULL) {
		free(mdl);
		return NULL;
	}
	for (int i = 0; i < 128; i++) {
		mdl->stt[i] = 0;
		mdl->rem[i] = INT_MAX;
	}
	mdl->itr  = 0;
	mdl->frq  = 0;
	mdl->dump = NULL;
	for (int i = 1; i < MAX_REAL; i++) {
		hsh_t idx = i;
		idx &= ((hsh_t)-1        >> (hsh_t)8);
		idx |= ((hsh_t)(128 - i) << (hsh_t)56);
		ftr_t *tmp = malloc(sizeof(ftr_t));
		memset(tmp, 0, sizeof(ftr_t));
		map_insert(mdl->ftrs, idx, tmp);
		mdl->real[i] = tmp;
	}
	return mdl;
}

/* mdl_newlbl:
 *   Build a new label object from the given string. This assume that the given
 *   string is non-empty and trimed if needed.
 *   The allocation of the object is done with care so a single free of the
 *   object will free all the associated memory leaving out the need for a
 *   dedicated function.
 */
static
lbl_t *mdl_newlbl(mdl_t *mdl, const char *str, int md) {
	assert(mdl != NULL && str != NULL && *str != '\0');
	// First pass: We just count the number of tokens in the input label so
	// we can allocate the label object in one block.
	int n = 1;
	for (int i = 0; str[i] != '\0'; i++)
		if (str[i] == '|')
			n++;
	// Now we can allocate the object in one malloc call and start filling
	// the object. There is no alignement problems here as only string are
	// put after the structure itself.
	lbl_t *lbl = malloc(sizeof(lbl_t) + sizeof(hsh_t) * n);
	if (lbl == NULL) {
		errno = ENOMEM;
		return NULL;
	}
	lbl->cnt = n;
	lbl->raw = ssp_string(mdl->ssp, str, md);
	// Next, we do the second pass on the string computing the hash values
	// of the tokens and filling the label object.
	for (int i = 0, l = 0, t = 0; ; i++) {
		if (str[i] == '|' || str[i] == '\0') {
			lbl->tok[t] = ssp_buffer(mdl->ssp, str + l, i - l, md);
			t++; l = i + 1;
		}
		if (str[i] == '\0')
			break;
	}
	return lbl;
}

/* mdl_maplbl:
 *   Map a label to its corresponding object in the given vocabulary. If the
 *   label is not already present in the table, create a new one and return it.
 *   This may only fail if the label is not present and the allocation fail.
 */
static
lbl_t *mdl_maplbl(mdl_t *mdl, map_t *voc, const char *str, int md) {
	assert(mdl != NULL && voc != NULL && str != NULL);
	// First search the vocabulary for the label. If it is found, we just
	// return the already existing label.
	const hsh_t hsh = hsh_string(str);
	lbl_t *lbl = map_find(voc, hsh);
	if (lbl != NULL)
		return lbl;
	// The label is not already in the table so create a new one and try to
	// insert it. We have to take some care here as another thread may have
	// inserted the same label in the mean time.
	lbl_t *tmp = mdl_newlbl(mdl, str, md);
	if (tmp == NULL)
		return NULL;
	lbl = map_insert(voc, hsh, tmp);
	if (lbl != tmp)
		free(tmp);
	return lbl;
}

/* mdl_map*:
 *   Small wrappers around mdl_maplbl for simple mapping in source or target
 *   vocabulary.
 */
static
lbl_t *mdl_mapsrc(mdl_t *mdl, const char *str) {
	assert(mdl != NULL && str != NULL);
	return mdl_maplbl(mdl, mdl->src, str, 0);
}
static
lbl_t *mdl_maptrg(mdl_t *mdl, const char *str) {
	assert(mdl != NULL && str != NULL);
	return mdl_maplbl(mdl, mdl->trg, str, 1);
}

static
ftr_t *mdl_addftr(mdl_t *mdl, int tag, int n, hsh_t hsh[n], int frq) {
	assert(mdl != NULL);
	assert(tag >= 0 && tag < 128);
	assert(n > 0 && hsh != NULL);
	// First, compute the feature identifier by combining the group tag and
	// the hash values in a single hash.
	hsh_t idx = hsh_buffer(hsh, sizeof(hsh_t) * n);
	idx &= ((hsh_t)-1  >> (hsh_t)8);
	idx |= ((hsh_t)tag << (hsh_t)56);
	// Search the table for the feature. If it is already present, just
	// return the associated object and increment frequency.
	ftr_t *ftr = map_find(mdl->ftrs, idx);
	if (ftr != NULL) {
		if (frq)
			atm_add(&ftr->frq, 1);
		return ftr;
	}
	// Check if the feature insertion is currently enabled for this tag, if
	// it is not the case, directly return NULL.
	if (mdl->itr < mdl->stt[tag] || mdl->itr >= mdl->rem[tag])
		return NULL;
	// Else, allocate a new object for the feature and try to insert it in
	// the map. On failure another thread have succeeded so just free our
	// new object and return the good one.
	ftr_t *tmp = malloc(sizeof(ftr_t));
	if (tmp == NULL) {
		errno = ENOMEM;
		return NULL;
	}
	memset(tmp, 0, sizeof(ftr_t));
	ftr = map_insert(mdl->ftrs, idx, tmp);
	if (ftr != tmp) {
		free(tmp);
		if (frq)
			atm_add(&ftr->frq, 1);
		return ftr;
	}
	// Else, insertion was a success so dump the feature if needed and
	// return it to the caller.
	if (mdl->dump != NULL) {
		fprintf(mdl->dump, "%016"PRIx64, idx);
		for (int i = 0; i < n; i++)
			fprintf(mdl->dump, " %016"PRIx64, hsh[i]);
		fprintf(mdl->dump, "\n");
	}
	if (frq)
		atm_add(&ftr->frq, 1);
	return ftr;
}

/* mdl_gettag:
 *   Return the tag of the given feature. This allow caller to retrieve tag
 *   parameter for any given feature.
 */
static
int mdl_gettag(ftr_t *ftr) {
	assert(ftr != NULL);
	return map_gethsh(ftr) >> (hsh_t)56;
}

/* mdl_next:
 *   Feature iterator. If [last] is NULL, return the first feature in the model,
 *   else return the next feature following [last] in the model.
 */
static
ftr_t *mdl_next(mdl_t *mdl, ftr_t *last) {
	assert(mdl != NULL);
	return map_next(mdl->ftrs, last);
}

/* mdl_remove:
 *   This act like the iterator function but also removing the given feature
 *   from the model.
 */
static
ftr_t *mdl_remove(mdl_t *mdl, ftr_t *last) {
	if (last == NULL)
		return mdl_next(mdl, last);
	const hsh_t hsh = map_gethsh(last);
	ftr_t *nxt = mdl_next(mdl, last);
	ftr_t *rem = map_remove(mdl->ftrs, hsh);
	if (rem != NULL)
		free(rem);
	return nxt;
}

/* mdl_shrink:
 *   Remove from the model all the features with a zero weight. For now, this
 *   code should only be called if no other threads are accesing the model.
 */
static
void mdl_shrink(mdl_t *mdl) {
	ftr_t *ftr = mdl_next(mdl, NULL);
	while (ftr != NULL) {
		if (ftr->x == 0.0)
			ftr = mdl_remove(mdl, ftr);
		else
			ftr = mdl_next(mdl, ftr);
	}
}

/* mdl_save:
 *   Save the model to the given file. The format is simple, one line per
 *   feature with the hash in hexadecimal followed by the feature value.
 */
static
int mdl_save(mdl_t *mdl, const char *fname) {
	assert(mdl != NULL && fname != NULL);
	FILE *file = fopen(fname, "w");
	if (file == NULL)
		return 0;
	ftr_t *ftr = mdl_next(mdl, NULL);
	while (ftr != NULL) {
		fprintf(file, "%016"PRIx64, map_gethsh(ftr));
		fprintf(file, " %.14f\n", ftr->x);
		ftr = mdl_next(mdl, ftr);
	}
	fclose(file);
	return 1;
}

static
int mdl_load(mdl_t *mdl, const char *fname) {
	assert(mdl != NULL && fname != NULL);
	FILE *file = fopen(fname, "r");
	if (file == NULL)
		return 0;
	while (!feof(file)) {
		hsh_t hsh; double wgh;
		if (fscanf(file, "%"PRIx64" %lf", &hsh, &wgh) != 2) {
			if (feof(file))
				break;
			fclose(file);
			return 0;
		}
		ftr_t *ftr = map_find(mdl->ftrs, hsh);
		if (ftr == NULL) {
			ftr = malloc(sizeof(ftr_t));
			if (ftr == NULL) {
				errno = ENOMEM;
				fclose(file);
				return 0;
			}
			memset(ftr, 0, sizeof(ftr_t));
			map_insert(mdl->ftrs, hsh, ftr);
		}
		ftr->x = wgh;
	}
	fclose(file);
	return 1;
}

/* mdl_stats:
 *   Display stats about total and active count of features. In verbose mode,
 *   stats per tag are displayed.
 */
static
void mdl_stats(mdl_t *mdl, int verb) {
	long tot[128], act[128];
	long t = 0, a = 0;
	memset(tot, 0, sizeof(tot));
	memset(act, 0, sizeof(act));
	ftr_t *ftr = mdl_next(mdl, NULL);
	while (ftr != NULL) {
		int tag = mdl_gettag(ftr);
		if (ftr->x != 0.0)
			act[tag]++, a++;
		tot[tag]++, t++;
		ftr = mdl_next(mdl, ftr);
	}
	if (verb) {
		for (int i = 0; i < 128; i++)
			if (tot[i] != 0)
				fprintf(stderr, "\ttag-%d=%ld/%ld\n",
					i, act[i], tot[i]);
	}
	fprintf(stderr, "\tftr=%ld/%ld\n", a, t);
}

/*******************************************************************************
 * Transducers
 ******************************************************************************/

typedef struct fst_s fst_t;
typedef struct arc_s arc_t;
typedef struct state_s state_t;
struct fst_s {
	int   acceptor;
	float mult;
	int   narcs, nstates;
	int   final;
	struct arc_s {
		// Input data: setup by the reader object, this is a direct
		// mapping of the input file.
		int      src,   trg;
		lbl_t   *ilbl, *olbl;
		double   wgh[MAX_REAL];
		// Arc features: setup by the generator, those are unigram
		// features who don't test the previous source segment.
		int      ucnt;    // =F
		ftr_t  **ulst;    // [F]
		// Gradient data: setup by the gradient itself but also used by
		// the decoder.
		double   psi;
		double   alpha;
		double   beta;
		// Decoder data: the decoder try to reuse as much as possible of
		// space used by the gradient but it need also those fields.
		int      eback;
		int      yback;
	} *arcs;
	struct state_s {
		int icnt, *ilst;
		int ocnt, *olst;
		int       **bcnt; // [NI][NO] --> NF
		ftr_t   ****blst; // [NI][NO][NF]
		double    **psi;
	} *states;
	int *s2t, *t2s;
	int     *raw_lst;
	void   **raw_ptr;
	int     *raw_cnt;
	ftr_t  **raw_ftr;
	double **raw_gptr;
	double  *raw_gval;
};

fst_t *fst_new(void) {
	fst_t *fst = malloc(sizeof(fst_t));
	fst->acceptor =  0;
	fst->narcs    =  0;
	fst->nstates  =  0;
	fst->final    = -1;
	fst->arcs     = NULL;
	fst->states   = NULL;
	fst->s2t      = NULL;
	fst->t2s      = NULL;
	fst->raw_lst  = NULL;
	fst->raw_ptr  = NULL;
	fst->raw_cnt  = NULL;
	fst->raw_ftr  = NULL;
	fst->raw_gptr = NULL;
	fst->raw_gval = NULL;
	return fst;
}

/* fst_addstates:
 *   Build the state list for the current FST. This work in multiple pass in
 *   order to keep code simple while using a single allocation.
 */
int fst_addstates(fst_t *fst) {
	assert(fst != NULL);
	if (fst->states != NULL &&fst->raw_lst != NULL)
		return 1;
	fst->states  = malloc(sizeof(state_t) * fst->nstates);
	fst->raw_lst = malloc(sizeof(int)     * fst->narcs * 2);
	if (fst->states == NULL || fst->raw_lst == NULL) {
		free(fst->raw_lst);
		free(fst->states);
		errno = ENOMEM;
		return 0;
	}
	for (int is = 0; is < fst->nstates; is++) {
		fst->states[is].icnt = 0;
		fst->states[is].ocnt = 0;
	}
	for (int ia = 0; ia < fst->narcs; ia++) {
		fst->states[fst->arcs[ia].trg].icnt++;
		fst->states[fst->arcs[ia].src].ocnt++;
	}
	int *raw = fst->raw_lst;
	for (int is = 0; is < fst->nstates; is++) {
		fst->states[is].ilst = raw;
		raw += fst->states[is].icnt;
		fst->states[is].icnt = 0;
		fst->states[is].olst = raw;
		raw += fst->states[is].ocnt;
		fst->states[is].ocnt = 0;
	}
	for (int ia = 0; ia < fst->narcs; ia++) {
		arc_t *a = &fst->arcs[ia];
		state_t *src = &fst->states[a->src];
		state_t *trg = &fst->states[a->trg];
		src->olst[src->ocnt++] = ia;
		trg->ilst[trg->icnt++] = ia;
	}
	return 1;
}

void fst_remstates(fst_t *fst) {
	free(fst->states);  fst->states  = NULL;
	free(fst->raw_lst); fst->raw_lst = NULL;
}

/* fst_toposort:
 *   Perform a topological sort of the states on the given FST and put the list
 *   of sorted states in the lst array. If the rev variable is true, the sort is
 *   performed from the final state instead of the initial one. This also check
 *   that their is a uniq extremum node and no cycles.
 */
int fst_toposort(const fst_t *fst, int *lst, int rev) {
	assert(fst != NULL && fst->states != NULL);
	assert(lst != NULL);
	const int N = fst->nstates;
	int deg[N];
	// First initialize the list of states and setup their degree according
	// to the direction of the sort.
	for (int n = 0; n < N; n++) {
		if (rev == 0)
			deg[n] = fst->states[n].icnt;
		else
			deg[n] = fst->states[n].ocnt;
		lst[n] = n;
	}
	// Next the main loop of the classical topological sort algorithm. We
	// search for state of degree 0, put them at the start of the list and
	// reduce the degrees.
	int done = 0;
	while (done < N) {
		// First put the states with no incoming edges at the start of
		// the list.
		int last = done;
		for (int n = done; n < N; n++) {
			if (deg[lst[n]] != 0)
				continue;
			const int tmp = lst[n];
			lst[n] = lst[last];
			lst[last] = tmp;
			last++;
		}
		// We check for some simple property we expect on the FST: a
		// single initial or final state and a lattice structure so no
		// cycles. The check are cheap so there is no reason to not
		// doing them.
		if (done == 0 && last != 1) {
			errno = EZEPFST;
			return 0;
		}
		if (last == done) {
			errno = EZEPFST;
			return 0;
		}
		// For each state in the current topological class, decrease the
		// degree of all the pointed state.
		for (int ni = done; ni < last; ni++) {
			state_t *s = fst->states;
			arc_t   *a = fst->arcs;
			int n = lst[ni];
			if (rev == 0)
				for (int i = 0; i < s[n].ocnt; i++)
					deg[a[s[n].olst[i]].trg]--;
			else
				for (int i = 0; i < s[n].icnt; i++)
					deg[a[s[n].ilst[i]].src]--;
		}
		done = last;
	}
	return 1;
}

/* fst_addsort:
 *   Build the topologicaly sorted list of arcs of the given FST in both
 *   directions. This also indirectly make a few checks on the FST: must have a
 *   single initial and final nodes and no cycles.
 */
int fst_addsort(fst_t *fst) {
	assert(fst != NULL && fst->states != NULL);
	if (fst->s2t != NULL && fst->t2s != NULL)
		return 1;
	const int A = fst->narcs, S = fst->nstates;
	fst->s2t = malloc(sizeof(int) * A);
	fst->t2s = malloc(sizeof(int) * A);
	if (fst->s2t == NULL || fst->t2s == NULL) {
		free(fst->s2t);
		free(fst->t2s);
		errno = ENOMEM;
		return 0;
	}
	int  lst[S];
	char flg[A]; memset(flg, 0, sizeof(char) * A);
	// First we sort in initial to final node. This is done by sorting the
	// nodes first and next using the outgoing edges lists to build the list
	// of sorted edges taking care of not duplicating them.
	fst_toposort(fst, lst, 0);
	for (int is = 0, p = 0; is < S; is++) {
		state_t *s = &fst->states[lst[is]];
		for (int ai = 0; ai < s->ocnt; ai++) {
			const int a = s->olst[ai];
			if (flg[a] == 1)
				continue;
			fst->s2t[p++] = a;
			flg[a] = 1;
		}
	}
	// Next we do the reverse, from final node to initial one using the same
	// principle.
	fst_toposort(fst, lst, 1);
	for (int is = 0, p = 0; is < S; is++) {
		state_t *s = &fst->states[lst[is]];
		for (int ai = 0; ai < s->icnt; ai++) {
			const int a = s->ilst[ai];
			if (flg[a] == 2)
				continue;
			fst->t2s[p++] = a;
			flg[a] = 2;
		}
	}
	return 1;
}

void fst_remsort(fst_t *fst) {
	free(fst->s2t); fst->s2t = NULL;
	free(fst->t2s); fst->t2s = NULL;
}

/*******************************************************************************
 * Dataset loader
 ******************************************************************************/

typedef struct dat_s dat_t;
struct dat_s {
	int     nfst;
	int     sfst;
	fst_t **fst;
};

/* dat_new:
 *   Create a new empty dataset ready to be populated with new FSTs. On error,
 *   return NULL and set errno.
 */
dat_t *dat_new(void) {
	dat_t *dat = malloc(sizeof(dat_t));
	if (dat == NULL) {
		errno = ENOMEM;
		return NULL;
	}
	dat->nfst = 0;
	dat->sfst = 0;
	dat->fst  = NULL;
	return dat;
}

/* dat_free:
 *   Free a dataset and all the FSTs object stored in it. No reference to the
 *   FSTs should be kept and used after this call. For simplicity this accept
 *   NULL as a valid dataset.
 */
void dat_free(dat_t *dat) {
	if (dat != NULL) {
		for (int i = 0; i < dat->nfst; i++)
			free(dat->fst[i]);
		free(dat->fst);
		free(dat);
	}
}

/* dat_parse:
 *   Parse a set of input line to build an FST object representing either a
 *   transducer or an acceptor. The labels are mapped using the given model.
 *   On error, this return NULL and set the errno appropriately.
 */
static
fst_t *dat_parse(char **lns, mdl_t *mdl) {
	voc_t *sts = NULL;
	int cnt = 0;
	while (lns[cnt] != NULL)
		cnt++;
	fst_t *fst = fst_new();
	fst->arcs = malloc(sizeof(arc_t) * cnt);
	if (fst->arcs == NULL) {
		errno = ENOMEM;
		goto error;
	}
	fst->states   = NULL;
	fst->acceptor = 0;
	fst->mult     = 0.0;
	fst->narcs    =   0;
	fst->nstates  =   0;
	fst->final    =  -1;
	// Next we parse the lines. A simple vocab is used to map the states
	// numbers so we are sure they are allocated as we want.
	sts = voc_new();
	if (sts == NULL)
		goto error;
	char *final = NULL;
	for (int i = 0; lns[i] != NULL; i++) {
		char *line = lns[i], *toks[4 + MAX_REAL];
		int ntoks = str_splitsp(line, 4 + MAX_REAL, toks);
		// First handle the case of empty and invalid lines. The only
		// possible error is three tokens for an FST as we ignore score
		// tokens and allow unused tokens.
		if (*line == '#')
			continue;
		if (ntoks == 0)
			continue;
		if (ntoks == 3) {
			errno = EZEPFMT;
			goto error;
		}
		// Now handle the simple case of final state, we just store the
		// token for future reference as if it is the first line, it
		// will be mapped to state 0 which is reseved for initial state.
		if (ntoks <= 2) {
			if (final != NULL) {
				errno = EZEPFMT;
				goto error;
			}
			final = toks[0];
			continue;
		}
		double wgh[MAX_REAL] = {};
		for (int i = 4; i < ntoks; i++)
			wgh[i - 4] = atof(toks[i]);
		// If we are here, the line define an arc and have the good
		// number of tokens, so it just remain to map the states and
		// labels and populate the arc array.
		const int src = voc_str2id(sts, toks[0]);
		const int trg = voc_str2id(sts, toks[1]);
		fst->nstates = max(fst->nstates, src + 1);
		fst->nstates = max(fst->nstates, trg + 1);
		lbl_t *ilbl = mdl_mapsrc(mdl, toks[2]);
		lbl_t *olbl = mdl_maptrg(mdl, toks[3]);
		const int ia = fst->narcs++;
		fst->arcs[ia].src  = src;
		fst->arcs[ia].trg  = trg;
		fst->arcs[ia].ilbl = ilbl;
		fst->arcs[ia].olbl = olbl;
		memcpy(fst->arcs[ia].wgh, wgh, sizeof(wgh));
	}
	if (final == NULL) {
		errno = EZEPFMT;
		goto error;
	}
	fst->final = voc_str2id(sts, final);
	// And finally, we just cleanup the vocab and return the parsed FST
	// object.
	voc_free(sts);
	return fst;
    error:
	if (sts != NULL)
		voc_free(sts);
	if (fst != NULL) {
		if (fst->arcs != NULL)
			free(fst->arcs);
		free(fst);
	}
	return NULL;
}

/* dat_load:
 *   Load the given input file in the dataset aither as a set of transducers or
 *   acceptors. If the multiplier is not zero, the default one is overriden with
 *   the providen one. On success, return 0, else return an approximate line
 *   number where the error was encountered.
 */
int dat_load(dat_t *dat, const char *fn, mdl_t *mdl, float mult, int ticks) {
	prg_t *prg = prg_new(ticks);
	assert(dat != NULL && fn != NULL);
	FILE *file = fopen(fn, "r");
	if (file == NULL)
		return 1;
	int ln = 1;
	prg_start(prg);
	while (!feof(file)) {
		char **lns = str_readeos(file);
		if (lns == NULL)
			break;
		fst_t *fst = dat_parse(lns, mdl);
		fst->mult = mult;
		for (int i = 0; lns[i] != NULL; i++, ln++)
			free(lns[i]);
		free(lns);
		if (fst == NULL)
			return ln;
		if (dat->nfst == dat->sfst) {
			int size = dat->sfst == 0 ? 128 : dat->sfst * 2;
			fst_t **tmp = realloc(dat->fst, sizeof(fst_t *) * size);
			if (tmp == NULL) {
				free(fst);
				errno = ENOMEM;
				return ln;
			}
			dat->sfst = size;
			dat->fst  = tmp;
		}
		dat->fst[dat->nfst++] = fst;
		prg_next(prg);
	}
	prg_end(prg);
	return 0;
}

/*******************************************************************************
 * Feature generator
 ******************************************************************************/

typedef struct pat_s pat_t;
typedef struct itm_s itm_t;
struct pat_s {
	hsh_t id;
	int cnt, tag;
	struct itm_s {
		int p1, p2; // first or second arc
		int s1, s2; // Source or target label
		int t1, t2; // Token number
	} itm[];
};
typedef struct gen_s gen_t;
struct gen_s {
	ssp_t  *ssp;
	int     nupat,   nbpat;
	pat_t **lupat, **lbpat;
	hsh_t   htrue,   hfalse;
	int     onref;
};

/* gen_new:
 *   Create a new empty feature generator. You must add at least one patterns or
 *   some bad things may happen.
 */
gen_t *gen_new(ssp_t *ssp, int onref) {
	gen_t *gen = malloc(sizeof(gen_t));
	gen->ssp    = ssp;
	gen->nupat  = 0;
	gen->lupat  = NULL;
	gen->nbpat  = 0;
	gen->lbpat  = NULL;
	gen->htrue  = ssp_string(ssp, "true", 0);
	gen->hfalse = ssp_string(ssp, "false", 0);
	gen->onref  = onref;
	return gen;
}

/* gen_free:
 *   Free all memory used by the pattern generator.
 */
void gen_free(gen_t *gen) {
	for (int i = 0; i < gen->nupat; i++)
		free(gen->lupat[i]);
	free(gen->lupat);
	for (int i = 0; i < gen->nbpat; i++)
		free(gen->lbpat[i]);
	free(gen->lbpat);
	free(gen);
}

/* gen_addpat:
 *   Add a new pattern in the generator, return false in case of syntactic
 *   error. This detect if the pattern in uni/bi-gram automaticaly and if it
 *   have a tag.
 */
int gen_addpat(gen_t *gen, const char *str) {
	int tag = 0, pos;
	if (sscanf(str, "%d:%n", &tag, &pos) == 1)
		str += pos;
	hsh_t id = 0;
	if (isalpha(*str)) {
		int len = 0;
		while (str[len] != '\0' && str[len] != ':')
			len++;
		if (str[len] == '\0')
			return 0;
		id = ssp_buffer(gen->ssp, str, len, 0);
		str += len + 1;
	}
	int cnt = 0;
	if (*str != '\0') {
		cnt = 1;
		for (int i = 0; str[i] != 0; i++)
			if (str[i] == ',')
				cnt++;
	}
	pat_t *pat = malloc(sizeof(pat_t) + sizeof(itm_t) * cnt);
	pat->tag = tag;
	pat->cnt = cnt;
	pat->id  = id;
	for (int i = 0; i < cnt; i++) {
		int p, t; char s;
		if (sscanf(str, "%d%c%d%n", &p, &s, &t, &pos) != 3)
			return 0;
		if (p < 0 || p > 1 || (s != 's' && s != 't'))
			return 0;
		pat->itm[i].p1 = p;
		pat->itm[i].s1 = s == 't';
		pat->itm[i].t1 = t;
		str += pos;
		if (*str != '=') {
			pat->itm[i].p2 = -1;
			pat->itm[i].s2 = -1;
			pat->itm[i].t2 = -1;
			str++;
			continue;
		}
		str = str + 1;
		if (sscanf(str, "%d%c%d%n", &p, &s, &t, &pos) != 3)
			return 0;
		if (p < 0 || p > 1 || (s != 's' && s != 't'))
			return 0;
		pat->itm[i].p2 = p;
		pat->itm[i].s2 = s == 't';
		pat->itm[i].t2 = t;
		str += pos + 1;
	}
	// Now, we check if the pattern is uni or bigram. We also take care of
	// fixing patterns referencing only the second arc.
	int p[2] = {0, 0};
	for (int i = 0; i < cnt; i++) {
		p[pat->itm[i].p1]++;
		if (pat->itm[i].p2 != -1)
			p[pat->itm[i].p2]++;
	}
	if (p[0] == 0) {
		for (int i = 0; i < cnt; i++) {
			pat->itm[i].p1--;
			pat->itm[i].p2--;
		}
		p[0] = p[1];
		p[1] = 0;
	}
	if (p[1] == 0) {
		gen->nupat += 1;
		gen->lupat = realloc(gen->lupat, sizeof(pat_t *) * gen->nupat);
		gen->lupat[gen->nupat - 1] = pat;
	} else {
		gen->nbpat += 1;
		gen->lbpat = realloc(gen->lbpat, sizeof(pat_t *) * gen->nbpat);
		gen->lbpat[gen->nbpat - 1] = pat;
	}
	return 1;
}

/* gen_ftralloc:
 *   This allocate in the given FST all the memory needed to store the features
 *   list. This include the multi-dimensional arrays used to acces the list and
 *   the raw block of memory which will be used to build the lists themselves.
 *   No other memory allocation should be needed.
 */
void gen_ftralloc(gen_t *gen, fst_t *fst) {
	assert(gen != NULL && fst != NULL);
	assert(fst->arcs != NULL && fst->states != NULL);
	if (fst->raw_ptr != NULL)
		return;
	const int S = fst->nstates;
	const int A = fst->narcs;
	// First pass on the FST: we compute the total size of the three raw
	// blocks of memory we will have to allocate. One for the pointers
	// arrays, one for the counts array, and one for the features arrays.
	int ptr = 0;
	int nu  = A, nb  = 0;
	for (int is = 0; is < S; is++) {
		state_t *nd = &fst->states[is];
		const int NI = nd->icnt;
		const int NO = nd->ocnt;
		ptr += NI * 2 + NI * NO;
		nb  += NI * NO;
	}
	const int cnt = nu + nb;
	const int ftr = nu * gen->nupat + nb * gen->nbpat;
	void  **rp = malloc(sizeof(void  *) * ptr);
	int    *rc = malloc(sizeof(int    ) * cnt);
	ftr_t **rf = malloc(sizeof(ftr_t *) * ftr);
	fst->raw_ptr = rp;
	fst->raw_cnt = rc;
	fst->raw_ftr = rf;
	// Second pass on the FST: we build the multi-dimensionnal arrays
	// structures. This setup all the needed stuff for the generation step
	// where we will populate the features lists.
	for (int ia = 0; ia < A; ia++) {
		arc_t *a = &fst->arcs[ia];
		a->ucnt = 0;
		a->ulst = rf;
		rf += gen->nupat;
	}
	for (int is = 0; is < S; is++) {
		state_t *nd = &fst->states[is];
		const int NI = nd->icnt;
		const int NO = nd->ocnt;
		nd->bcnt = (int     **)rp; rp += NI;
		nd->blst = (ftr_t ****)rp; rp += NI;
		for (int ni = 0; ni < NI; ni++) {
			nd->bcnt[ni] = (int     *)rc; rc += NO;
			nd->blst[ni] = (ftr_t ***)rp; rp += NO;
			for (int no = 0; no < NO; no++) {
				nd->bcnt[ni][no] = 0;
				nd->blst[ni][no] = rf;
				rf += gen->nbpat;
			}
		}
	}
}

/* gen_remftr:
 *   Free all memory used to store the features lists. This should be run over
 *   all FST before removing features in the model in order to be sure no
 *   reference are kept over the removed features.
 */
void gen_remftr(fst_t *fst) {
	free(fst->raw_ptr); fst->raw_ptr = NULL;
	free(fst->raw_cnt); fst->raw_cnt = NULL;
	free(fst->raw_ftr); fst->raw_ftr = NULL;
}

/* gen_get:
 *   Return the hash value for the given item taken from the label set. This
 *   take care of handling the equality features.
 */
static inline
hsh_t gen_get(gen_t *gen, itm_t *itm, lbl_t *lbl[]) {
	hsh_t h1 = lbl[itm->p1 * 2 + itm->s1]->tok[itm->t1];
	if (itm->p2 < 0)
		return h1;
	hsh_t h2 = lbl[itm->p2 * 2 + itm->s2]->tok[itm->t2];
	if (h1 == h2) return gen->htrue;
	else          return gen->hfalse;
}

/* gen_uftr:
 *   Generate the unigram feature list for the given label array.
 */
static
int gen_uftr(gen_t *gen, mdl_t *mdl, lbl_t *lbl[], ftr_t *lst[], int frq) {
	int cnt = 0;
	for (int i = 0; i < gen->nupat; i++) {
		pat_t *pat = gen->lupat[i];
		hsh_t hsh[pat->cnt + 1];
		hsh[0] = pat->id;
		int off = hsh[0] != 0;
		for (int j = 0; j < pat->cnt; j++)
			hsh[j + off] = gen_get(gen, &pat->itm[j], lbl);
		ftr_t *ftr;
		ftr = mdl_addftr(mdl, pat->tag, pat->cnt + off, hsh, frq);
		if (ftr != NULL)
			lst[cnt++] = ftr;
	}
	return cnt;
}

/* gen_bftr:
 *   Generate the bigram feature list for the given label array.
 */
static
int gen_bftr(gen_t *gen, mdl_t *mdl, lbl_t *lbl[], ftr_t *lst[], int frq) {
	int cnt = 0;
	for (int i = 0; i < gen->nbpat; i++) {
		pat_t *pat = gen->lbpat[i];
		hsh_t hsh[pat->cnt + 1];
		hsh[0] = pat->id;
		int off = hsh[0] != 0;
		for (int j = 0; j < pat->cnt; j++)
			hsh[j + off] = gen_get(gen, &pat->itm[j], lbl);
		ftr_t *ftr;
		ftr = mdl_addftr(mdl, pat->tag, pat->cnt + off, hsh, frq);
		if (ftr != NULL)
			lst[cnt++] = ftr;
	}
	return cnt;
}

/* gen_addftr:
 *   Add features list on the given FST. This can be costly but also take quite
 *   some memory, so there is a tradeoff in generating them at each iterations.
 *   Remember that they should be regenerated if some features are removed from
 *   the model.
 */
void gen_addftr(gen_t *gen, mdl_t *mdl, fst_t *fst) {
	int frq = 0;
	if (fst->mult < 0 &&  gen->onref) frq = 1;
	if (fst->mult > 0 && !gen->onref) frq = 1;
	gen_ftralloc(gen, fst);
	for (int ia = 0; ia < fst->narcs; ia++) {
		arc_t  *a  = &fst->arcs[ia];
		lbl_t *lbl[2] = {a->ilbl, a->olbl};
		a->ucnt = gen_uftr(gen, mdl, lbl, a->ulst, frq);
	}
	for (int is = 0; is < fst->nstates; is++) {
		state_t *s = &fst->states[is];
		for (int ii = 0; ii < s->icnt; ii++) {
		for (int io = 0; io < s->ocnt; io++) {
			arc_t *ai = &fst->arcs[s->ilst[ii]];
			arc_t *ao = &fst->arcs[s->olst[io]];
			lbl_t *lbl[4] = {
				ai->ilbl, ai->olbl,
				ao->ilbl, ao->olbl};
			ftr_t **lst = s->blst[ii][io];
			s->bcnt[ii][io] = gen_bftr(gen, mdl, lbl, lst, frq);
		}
		}
	}
}


/*******************************************************************************
 * Gradient computer
 ******************************************************************************/

typedef struct grd_s grd_t;
struct grd_s {
	int    nth;
	int    cache;
	double fx;
	dat_t *dat;
	gen_t *gen;
	mdl_t *mdl;
	prg_t *prg;
	int    idx;
};

/* grd_new:
 *   Setup a new gradient computer. By default, it works with a single thread bu
 *   this can be changed.
 */
static
grd_t *grd_new(mdl_t *mdl, gen_t *gen, dat_t *dat) {
	grd_t *grd = malloc(sizeof(grd_t));
	grd->nth = 1;
	grd->dat = dat;
	grd->gen = gen;
	grd->mdl = mdl;
	return grd;
}

/* grd_free:
 *   Free all memory associated with the given gradient.
 */
static
void grd_free(grd_t *grd) {
	free(grd);
}

/* grd_addspc:
 *   Allocate memory in the FST for all temporarie variables the gradient need
 *   to store during the computation.
 */
static
void grd_addspc(fst_t *fst) {
	if (fst->raw_gptr != NULL && fst->raw_gval != NULL)
		return;
	// We use the same strategy than for allocating features lists. First
	// pass on the FST we count how much data we have to allocate and
	// prepare big buffer that we will split ourselve.
	int np = 0, nv = 0;
	for (int is = 0; is < fst->nstates; is++) {
		state_t *s = &fst->states[is];
		np += s->icnt;
		nv += s->icnt * s->ocnt;
	}
	double **rp = malloc(sizeof(double *) * np);
	double  *rv = malloc(sizeof(double  ) * nv);
	memset(rv, 0, sizeof(double) * nv);
	fst->raw_gptr = rp;
	fst->raw_gval = rv;
	// Now we make a second pass on the data to build the multi-dimensional
	// arrays from the blocks we have prepared just before.
	for (int is = 0; is < fst->nstates; is++) {
		state_t *s = &fst->states[is];
		s->psi = (double **)rp; rp += s->icnt;
		for (int i = 0; i < s->icnt; i++)
			s->psi[i] = rv, rv += s->ocnt;
	}
}

/* grd_remspc:
 *   Free all memory used to compute the gradient. If features are removed from
 *   the fst, this should also be to keep things in sync.
 */
static
void grd_remspc(fst_t *fst) {
	free(fst->raw_gptr); fst->raw_gptr = NULL;
	free(fst->raw_gval); fst->raw_gval = NULL;
}

/* grd_dopsi:
 *   We first have to compute the Ψ_e(y',y,x) weights defined as
 *       Ψ_e(y',y,x) = \exp(   ∑_k θ_k f_k(y,x_e)
 *                           + ∑_k θ_k f_k(y',y,x_e) )
 *   We split this computation in three distinct parts for the tree kinds of
 *   features we handle. They are stored in the two *psi tables and will be
 *   grouped together later.
 *   To avoid numerical problems we will do all the computations in log-space
 *   so, here, we just skip the exponential and just compute the sums.
 */
static
void grd_dopsi(const mdl_t *mdl, fst_t *fst) {
	for (int ia = 0; ia < fst->narcs; ia++) {
		arc_t *a = &fst->arcs[ia];
		double sum = 0.0;
		for (int f = 0; f < a->ucnt; f++)
			sum += a->ulst[f]->x;
		a->psi = sum + a->wgh[0];
		for (int i = 1; i < MAX_REAL; i++) {
			// FIXME: hack Nicolas
			// Only use the real features it they should be included
			// in the model
			if (mdl->stt[mdl_gettag(mdl->real[i])] <= mdl->itr)
				a->psi += mdl->real[i]->x * a->wgh[i];
		}
	}
	for (int is = 0; is < fst->nstates; is++) {
		const state_t *s = &fst->states[is];
		for (int ni = 0; ni < s->icnt; ni++) {
		for (int no = 0; no < s->ocnt; no++) {
			double sum = 0.0;
			for (int f = 0; f < s->bcnt[ni][no]; f++)
				sum += s->blst[ni][no][f]->x;
			s->psi[ni][no] = sum;
		}
		}
	}
}

/* logsum:
 *  Compute log(exp(a) + exp(b)) while minimizing precision loss. Main use is
 *  computing the addition of two values in log-space.
 */
static
double logsum(const double a, const double b) {
	if (a == -DBL_MAX) return b;
	else if (a > b)    return a + log(1 + exp(b - a));
	else               return b + log(1 + exp(a - b));
}

/* grd_fwdbwd:
 *   Now, we go for the forward-backward algorithm. Both pass are similar except
 *   they walk the lattice in different order. The forward pass recursion is
 *   defined by:
 *       | α_1(y) = Ψ_0(y,x)
 *       | α_n(y) = ∑_{y'} α_{t-1}(y') * Ψ_e(y',y,x)
 *   and the backward one by:
 *       | β_N    (y') = 1
 *       | β_{n-1}(y') = ∑_{y} β_t(y) * Ψ_e(y',y,x)
 *   As we do the computations in log-space, the products are replaced by sums
 *   and the sums by logsums.
 *   This is where we have to sum the two components of the psi function.
 */
static
void grd_fwdbwd(fst_t *fst) {
	const int A = fst->narcs;
	// The forward recurence: We walk over all the arcs in topological
	// order so we are sure that all incoming arcs of the source state of
	// the current arc are processed before the current arc.
	int *s2t = fst->s2t;
	for (int io = 0, o = s2t[0]; io < A; o = s2t[++io]) {
		arc_t   *ao = &fst->arcs[o];
		state_t *st = &fst->states[ao->src];
		// We first handle the case where there is no incoming arc in
		// the source state. This is the initial case of the recursion
		// so we just copy the local psi to alpha.
		if (ao->src == 0) {
			ao->alpha = ao->psi;
			continue;
		}
		// Else, there is some incoming arcs so we have to perform the
		// sum of all incomming arcs combined with the current one. The
		// first step is to find the index of the current arc in the
		// outgoing list of the state.
		int no = 0;
		for ( ; no < st->ocnt; no++)
			if (st->olst[no] == o)
				break;
		// Next we process all incoming arcs and sum there contribution
		// to compute the alpha value using the recursion.
		ao->alpha = -DBL_MAX;
		for (int ni = 0; ni < st->icnt; ni++) {
			const arc_t *ai = &fst->arcs[st->ilst[ni]];
			double v = ao->psi + st->psi[ni][no] + ai->alpha;
			ao->alpha = logsum(ao->alpha, v);
		}
	}
	// The backward recurence: it is done exactly as the forward one except
	// that we process the state in topological order from the final state
	// and we look for the outgoing arcs of the target state.
	int *t2s = fst->t2s;
	for (int ii = 0, i = t2s[0]; ii < A; i = t2s[++ii]) {
		arc_t   *ai = &fst->arcs[i];
		state_t *st = &fst->states[ai->trg];
		// Same as before: we first check for cases with no outgoing
		// arcs.
		if (ai->trg == fst->final) {
			ai->beta = 0.0;
			continue;
		}
		// Else we first search the position of the current arc in the
		// incoming list.
		int ni = 0;
		for ( ; ni < st->icnt; ni++)
			if (st->ilst[ni] == i)
				break;
		// And finally the recurence itself like in the forward pass.
		ai->beta = -DBL_MAX;
		for (int no = 0; no < st->ocnt; no++) {
			const arc_t *ao = &fst->arcs[st->olst[no]];
			double v = ao->psi + st->psi[ni][no] + ao->beta;
			ai->beta = logsum(ai->beta, v);
		}
	}
}

/* grd_doupd:
 *   The normalization constant can be computed with
 *       Z_θ = ∑_y α_n(y) β_n(y)
 *   either at initial or final node. The last one is the easiest as the β_n(y)
 *   are all equal to 1.0 and so can be ignored. We also have to be carefull
 *   here to do the computation in log-space.
 *   Now, we have all we need to compute the gradient of the negative log-
 *   likelihood
 *       ∂-L(θ)      ∑_n ∑_{y}    f_k(y,x_n)    p_θ(y_n=y|x)
 *       ------ =  ±
 *        ∂θ_k       ∑_e ∑_{y',y} f_k(y',y,x_e) p_θ(y_e.s=y',y_e.t=y|x)
 *   This is the expectation of f_k under the model distribution, this is the
 *   true expectation over the model in case of an hypothesis lattice, and the
 *   empirical distribution in case of reference lattice. This is where we use
 *   all the previous computations. The probabilities are given by:
 *       p_θ(y_n=y|x)            = α_n(y) β_n(y) / Z_θ
 *       p_θ(y_e.s=y',y_e.t=y|x) = α_e.s(y') Ψ_e(y',y,x) β_e.t(y) / Z_θ
 */
static
double grd_doupd(mdl_t *mdl, fst_t *fst) {
	const int A = fst->narcs;
	const int S = fst->nstates;
	const double mul = fst->mult;
	// Computing the normalization constant is quite simple, we just have to
	// take the sum of all the alpha values of the edges pointing to the
	// final node. We don't have to care multiplying by the beta values as
	// they should be equal to 1. The only trickery is that we have to
	// perform the computation in log-space.
	double Z = -DBL_MAX;
	for (int ia = 0; ia < A; ia++) {
		const arc_t *a = &fst->arcs[ia];
		if (a->trg == fst->final)
			Z = logsum(Z, a->alpha);
	}
	// Next we have to compute the probability of the edge unigrams features
	// who are the most simple ones. The expectation of them is just the
	// product of the corresponding alpha and beta values divided by the
	// normalization constant. (also computed in log)
	for (int ia = 0; ia < A; ia++) {
		arc_t *a = &fst->arcs[ia];
		const double ex = exp(-Z + a->alpha + a->beta);
		for (int f = 0; f < a->ucnt; f++)
			atm_inc(&a->ulst[f]->g, ex * mul);
		for (int i = 1; i < MAX_REAL; i++)
			atm_inc(&mdl->real[i]->g, ex * a->wgh[i] * mul);
	}
	// The node features are a bit more complex as they involve two edges.
	// We loop over all nodes and for each of them loop over all possible
	// combination of an incoming and an outgoing edge.
	for (int is = 0; is < S; is++) {
		const state_t *s = &fst->states[is];
		for (int ni = 0; ni < s->icnt; ni++) {
		for (int no = 0; no < s->ocnt; no++) {
			const arc_t *ai = &fst->arcs[s->ilst[ni]];
			const arc_t *ao = &fst->arcs[s->olst[no]];
			// Now, for each of them we have to compute the
			// expectation which is a bit more complicated as we
			// have to add the contribution of the current edge.
			int     nbf = s->bcnt[ni][no];
			ftr_t **lbf = s->blst[ni][no];
			double ex = exp(-Z + ai->alpha + ao->beta
			                   + ao->psi + s->psi[ni][no]);
			for (int f = 0; f < nbf; f++)
				atm_inc(&lbf[f]->g, ex * mul);
		}
		}
	}
	return mul * Z;
}

static
void *grd_worker(void *ud) {
	grd_t *grd = ud;
	double fx = 0.0;
	while (1) {
		int id = atm_add(&grd->idx, 1) - 1;
		if (id >= grd->dat->nfst)
			break;
		fst_t *fst = grd->dat->fst[id];
		fst_addstates(fst);
		fst_addsort(fst);
		gen_addftr(grd->gen, grd->mdl, fst);
		grd_addspc(fst);
		grd_dopsi(grd->mdl, fst);
		grd_fwdbwd(fst);
		fx += grd_doupd(grd->mdl, fst);
		if (grd->cache < 4)
			grd_remspc(fst);
		if (grd->cache < 3)
			gen_remftr(fst);
		if (grd->cache < 2)
			fst_remsort(fst);
		if (grd->cache < 1)
			fst_remstates(fst);
		prg_next(grd->prg);
	}
	atm_inc(&grd->fx, fx);
	return NULL;
}

/* grd_compute:
 *   Compute the gradient given the current value of the features in the model
 *   and set the g field of all of them. This expect the g field to be cleared
 *   before the call.
 */
static
double grd_compute(grd_t *grd) {
	grd->prg = prg_new(grd->dat->nfst / 49);
	grd->idx = 0;
	grd->fx  = 0.0;
	prg_start(grd->prg);
	if (grd->nth == 1) {
		grd_worker(grd);
	} else {
		thread_t thrd[grd->nth];
		for (int n = 0; n < grd->nth; n++)
			thread_spawn(&thrd[n], grd_worker, grd);
		for (int n = 0; n < grd->nth; n++)
			thread_join(thrd[n]);
	}
	prg_end(grd->prg);
	return grd->fx;
}

/*******************************************************************************
 * Optimizer
 *
 *   Implement the model optimizer using resilient back-propagation [1], this
 *   use all the previous thing to perform the actual model training in an
 *   efficient way.
 *   In order to optimize a model, the optimize function should be called until
 *   convergence. It is the responsibility of the caller code to determine when
 *   the model is fully trained.
 *
 * [1] A direct adaptive method for faster backpropagation learning: The RPROP
 *     algorithm, Martin Riedmiller and Heinrich Braun, IEEE International
 *     Conference on Neural Networks, San Francisco, USA, 586-591, March 1993.
 ******************************************************************************/

typedef struct rbp_s rbp_t;
struct rbp_s {
	double rho1[128];
	double rho2[128];
	double rho3[128];
	double stpinc;
	double stpdec;
	double stpmin;
	double stpmax;
};

static
rbp_t *rbp_new(void) {
	rbp_t *rbp = malloc(sizeof(rbp_t));
	if (rbp == NULL)
		return NULL;
	rbp->rho1[0] = rbp->rho2[0] = rbp->rho3[0] = 0.0;
	for (int i = 1; i < 128; i++)
		rbp->rho1[i] = rbp->rho2[i] = rbp->rho3[i] = -1.0;
	rbp->stpinc = 1.2;
	rbp->stpdec = 0.5;
	rbp->stpmin = 1e-8;
	rbp->stpmax = 50.0;
	return rbp;
}

static
void rbp_free(rbp_t *rbp) {
	free(rbp);
}

/* rbp_step:
 *   Perform one step of the resilient back-propagation algorithm including
 *   computation of the gradient and applying the regularization. Return the
 *   value of the objective function before the optimization step. (computing
 *   the new value would require a second computation which is a lot too costly)
 */
static
void rbp_step(rbp_t *rbp, mdl_t *mdl, double ll) {
	assert(rbp != NULL && mdl != NULL);
	prg_t *prg = prg_new(mdl->ftrs->count / 49);
	double nx  = 0.0, ng  = 0.0, nd = 0.0;
	double fx  = ll;
	ftr_t *ftr = NULL;
	prg_start(prg);
	while (1) {
		ftr = mdl_next(mdl, ftr);
	    next:
		if (ftr == NULL)
			break;
		const int tag = mdl_gettag(ftr);
		// Check if we should remove the feature either for to low freq
		// or for having a zero weight.
		// FIXME: Hack Nico : We also want to ignore dense features
		// that should not be included in the model
		if (ftr->x == 0.0 && mdl->rem[tag] <= mdl->itr) {
			ftr = mdl_remove(mdl, ftr);
			goto next;
		} else if (ftr->frq < mdl->frq) {
			ftr = mdl_remove(mdl, ftr);
			goto next;
		} else if (mdl->stt[tag] > mdl->itr) {
			ftr = mdl_next(mdl, ftr);
			goto next;
		}
		// We detect new feature with their step size being zero and
		// initialize them. The model should have set all fields to zero
		// so we just have to setup the step size.
		if (ftr->stp == 0.0)
			ftr->stp = 0.1;
		// We retrieve the feature tag and regularization parameter and
		// update the l1 & l2 norm of the model.
		const double rho1 = rbp->rho1[tag];
		const double rho2 = rbp->rho2[tag];
		const double rho3 = rbp->rho3[tag];
		ftr->g += rho2 * ftr->x;
		fx += rho2 * ftr->x * ftr->x / 2.0;
		fx += rho1 * fabs(ftr->x);
		fx += rho3 * ftr->frq * fabs(ftr->x);
		// First step is to project the gradient in the current orthant
		// to ensure derivability.
		const double ar = rho1 + rho3 * ftr->frq;
		double pg = ftr->g;
		if (ar != 0) {
			     if (ftr->x < -EPSILON) pg -= ar;
			else if (ftr->x >  EPSILON) pg += ar;
			else if (ftr->g < -ar     ) pg += ar;
			else if (ftr->g >  ar     ) pg -= ar;
			else                        pg  = 0.0;
		}
		// Next we adjust the step depending on the new and previous
		// gradient sign.
		const double sgn = ftr->gp * pg;
		if (sgn < -EPSILON)
			ftr->stp = max(ftr->stp * rbp->stpdec, rbp->stpmin);
		else if (sgn > EPSILON)
			ftr->stp = min(ftr->stp * rbp->stpinc, rbp->stpmax);
		// And we update the weight. If gradient sign changed, we take
		// back the previous update, else we make one step in gradient
		// direction and project back in orthant.
		if (sgn < 0.0) {
			ftr->x -= ftr->dlt;
			ftr->g  = 0.0;
		} else {
			     if (pg < -EPSILON) ftr->dlt =  ftr->stp;
			else if (pg >  EPSILON) ftr->dlt = -ftr->stp;
			else                    ftr->dlt = 0.0;
			if (rho1 != 0.0 && ftr->dlt * pg >= 0.0)
				ftr->dlt = 0.0;
			ftr->x += ftr->dlt;
		}
		// Finally, prepare the feature for the next iteration. We save
		// the current gradient and clear it.
		nx += fabs(ftr->x);
		ng += fabs(ftr->g);
		nd += fabs(ftr->dlt);
		ftr->frq = 0;
		ftr->gp  = ftr->g;
		ftr->g   = 0.0;
		prg_next(prg);
	}
	prg_end(prg);
	fprintf(stderr, "\tll=%.2f", -ll);
	fprintf(stderr, " fx=%.2f",  fx);
	fprintf(stderr, " |x|=%.2f",   nx);
	fprintf(stderr, " |g|=%.2f",   ng);
	fprintf(stderr, " |d|=%.2f\n", nd);
}

/*******************************************************************************
 * Decoder
 ******************************************************************************/

/* dec_forward:
 *   The Viterbi forward step. This is the same than the gradient forward step
 *   with the difference that we work in the tropical semi-ring instead of the
 *   log one.
 */
static
void dec_forward(fst_t *fst) {
	int *s2t = fst->s2t;
	for (int io = 0, o = s2t[0]; io < fst->narcs; o = s2t[++io]) {
		arc_t   *ao = &fst->arcs[o];
		state_t *nd = &fst->states[ao->src];
		// We first handle the case where there is no incoming arc in
		// the source state. This is the initial case of the recursion
		// so we just copy the local psi array to alpha.
		if (ao->src == 0) {
			ao->alpha = ao->psi;
			continue;
		}
		// Else, there is some incomming arcs so we have to perform the
		// sum of all incomming arcs combined with the current one. The
		// first step is to find the index of the current arc in the
		// outgoing list of the state and to initialize the alpha array
		// to "log(0)" for the sum.
		int no = 0;
		for ( ; no < nd->ocnt; no++)
			if (nd->olst[no] == o)
				break;
		// Next we process all incoming arcs and search the one with
		// the maximum score. This is the main difference with the
		// gradient forward step where we sum over all of them.
		ao->alpha = -DBL_MAX;
		for (int ni = 0; ni < nd->icnt; ni++) {
			const arc_t *ai = &fst->arcs[nd->ilst[ni]];
			double v = ao->psi + nd->psi[ni][no] + ai->alpha;
			if (v > ao->alpha) {
				ao->alpha = v;
				ao->eback = nd->ilst[ni];
				ao->yback = 0;
			}
		}
	}
}

/* dec_backtrack:
 *   The equivalent of the backward step of the gradient for Viterbi decoding.
 *   Here we don't have to compute the scores, we just follow the best path
 *   found in the previous step to find the full path. The path is stored in
 *   reverse order in the arrays [eds] and [lbl]. The former receive the edge
 *   number of the path, while the second receive the hypothesis number for each
 *   edge.
 */
static
int dec_backtrack(fst_t *fst, lbl_t *out[][2]) {
	const int E = fst->narcs;
	// First find the end point of the best path. We search the label with
	// the best score in all the edge pointing to the final node.
	double bst = -DBL_MAX;
	int ei = 0;
	for (int e = 0; e < E; e++) {
		const arc_t *ed = &fst->arcs[e];
		if (ed->trg != fst->final)
			continue;
		if (ed->alpha > bst) {
			bst = ed->alpha;
			ei  = e;
		}
	}
	int pos = 1;
	out[0][0] = fst->arcs[ei].ilbl;
	out[0][1] = fst->arcs[ei].olbl;
	// Next we follow the backtrack pointers until we reach the starting
	// point of the lattice filling the output array as we go.
	arc_t *ed = &fst->arcs[ei];
	while (ed->src != 0) {
		ei = ed->eback;
		out[pos  ][0] = fst->arcs[ei].ilbl;
		out[pos++][1] = fst->arcs[ei].olbl;
		ed = &fst->arcs[ei];
	}
	return pos;
}

static
int dec_dsmap(voc_t *voc, int n1, int n2) {
	char buffer[64];
	snprintf(buffer, sizeof(buffer), "%d-%d", n1, n2);
	return voc_str2id(voc, buffer) + 2;
}

static
void dec_dumpspc(fst_t *fst, ssp_t *ssp, FILE *file) {
	voc_t *voc = voc_new();
	state_t *sti = &fst->states[0];
	for (int no = 0; no < sti->ocnt; no++) {
		const int eo = sti->olst[no];
		const arc_t *ed = &fst->arcs[eo];
		hsh_t hi = fst->arcs[eo].ilbl->raw;
		hsh_t ho = fst->arcs[eo].olbl->raw;
		const char *ilbl = ssp_get(ssp, hi);
		const char *olbl = ssp_get(ssp, ho);
		const double sc  = ed->psi;
		const int    trg = dec_dsmap(voc, eo, 0);
		fprintf(file, "0\t%d\t", trg);
		fprintf(file, "%s\t%s\t", ilbl, olbl);
		fprintf(file, "%f\n", sc);
	}
	const int S = fst->nstates;
	for (int s = 0; s < S; s++) {
		const state_t *nd = &fst->states[s];
		for (int ni = 0; ni < nd->icnt; ni++) {
		for (int no = 0; no < nd->ocnt; no++) {
			const int ei = nd->ilst[ni];
			const int eo = nd->olst[no];
			const arc_t *ed = &fst->arcs[eo];
			hsh_t hi = fst->arcs[eo].ilbl->raw;
			hsh_t ho = fst->arcs[eo].olbl->raw;
			const char *ilbl = ssp_get(ssp, hi);
			const char *olbl = ssp_get(ssp, ho);
			const double sc  = nd->psi[ni][no] + ed->psi;
			const int src = dec_dsmap(voc, ei, 0);
			const int trg = dec_dsmap(voc, eo, 0);
			fprintf(file, "%d\t%d\t", src, trg);
			fprintf(file, "%s\t%s\t", ilbl, olbl);
			fprintf(file, "%f\n", sc);
		}
		}
	}
	state_t *stf = &fst->states[fst->final];
	for (int ni = 0; ni < stf->icnt; ni++) {
		const int ei = stf->ilst[ni];
			const int src = dec_dsmap(voc, ei, 0);
			fprintf(file, "%d\t1\t<eps>\t0.0\n", src);
	}
	fprintf(file, "1\nEOS\n");
	voc_free(voc);
}

static
void dec_decode(mdl_t *mdl, ssp_t *ssp, gen_t *gen, dat_t *dat, FILE *file,
		int spc) {
	prg_t *prg = prg_new(1000);
	prg_start(prg);
	for (int i = 0; i < dat->nfst; i++) {
		fst_t *fst = dat->fst[i];
		fst_addstates(fst);
		fst_addsort(fst);
		gen_addftr(gen, mdl, fst);
		grd_addspc(fst);
		grd_dopsi(mdl, fst);
		if (spc == 0) {
			dec_forward(fst);
			lbl_t *out[fst->narcs][2];
			int cnt = dec_backtrack(fst, out);
			for (int i = cnt - 1; i >= 0; i--) {
				hsh_t ihsh = map_gethsh(out[i][0]);
				hsh_t ohsh = map_gethsh(out[i][1]);
				fprintf(file, "%s@", ssp_get(ssp, ihsh));
				fprintf(file, "%s ", ssp_get(ssp, ohsh));
			}
			fprintf(file, "\n");
		} else {
			dec_dumpspc(fst, ssp, file);
		}
		grd_remspc(fst);
		gen_remftr(fst);
		fst_remsort(fst);
		fst_remstates(fst);
		prg_next(prg);
	}
	prg_end(prg);
	prg_free(prg);
}

/*******************************************************************************
 * Command line parsing
 *
 *   Implement a simple and generic command line argument parsing allowing a
 *   wide variety of options type and a simple way to define new switch. This
 *   mean that adding a new switch is very easy and so all there is no excuse to
 *   not make configurable anything that can be.
 ******************************************************************************/

typedef struct arg_s arg_t;
struct arg_s {
	char  type;
	char *dshort;
	char *dlong;
	void *action;
	void *data;
};
typedef void (* arg_func0_t)(void *ud, char *cmd);
typedef void (* arg_func1_t)(void *ud, char *cmd, char *arg);

/* arg_parse:
 *   Parse command line argument according to the given definitions and remove
 *   them from the arg-list. After this call only non-switch arguments remain in
 *   the list.
 */
static
void arg_parse(arg_t *def, int *argc, char *argv[]) {
	assert(def != NULL && argc != NULL && argv != NULL);
	int pos = 0, rem = 0;
	while (pos < *argc) {
		char *arg = argv[pos++];
		assert(arg != NULL);
		// If we encounter the switch terminator, we just move down the
		// remaining arguments and return. Else, if the argument is not
		// an option, we move it down and go to the next one.
		if (!strcmp(arg, "--")) {
			while (pos < *argc)
				argv[rem++] = argv[pos++];
			break;
		} else if (arg[0] != '-') {
			argv[rem++] = arg;
			continue;
		}
		// Else we search the argument in the switch list and check its
		// argument if one is needed.
		int id = 0;
		for ( ; def[id].type != 0; id++) {
			if (!strcmp(arg, def[id].dshort))
				break;
			if (!strcmp(arg, def[id].dlong))
				break;
		}
		if (def[id].type == 0)
			fatal("unknown switch %s", arg);
		arg_t *itm = &def[id];
		char *val = NULL;
		if (itm->type != 'b' && itm->type != '0') {
			if (pos == *argc)
				fatal("missing argument for %s", arg);
			val = argv[pos++];
		}
		// Now, look the type of the switch and handle it with all the
		// error checking needed.
		const char tp = itm->type;
		switch (tp) {
			// Boolean: there is no arguments, we just have to turn
			// on the switch.
			case 'b': {
				*((int *)itm->action) = 1;
				break;
			}
			// Numbers: Handle both signed and unsigned value with
			// the same code with just an additional check for
			// unsigned.
			case 'i': case 'u': {
				char *end = NULL;
				const int nb = (int)strtol(val, &end, 10);
				if (*end != '\0' || (tp == 'u' && nb < 0))
					fatal("invalid argument for %s", arg);
				*((int *)itm->action) = nb;
				break;
			}
			// Floating point values: they are handled similarily
			// than the integers values.
			case 'f': case 'p': {
				char *end = NULL;
				const double fp = strtod(val, &end);
				if (*end != '\0' || (tp == 'p' && fp < 0))
					fatal("invalid argument for %s", arg);
				*((double *)itm->action) = fp;
				break;
			}
			// Single string: this is quite simple but if the same
			// switch is provided multiple time, only the last value
			// is kept.
			case 's': {
				*((char **)itm->action) = val;
				break;
			}
			// List of string: We have to add a new string at the
			// end of the NULL terminated list.
			case 'S': {
				size_t n = 0;
				char ***lst = itm->action;
				if (*lst != NULL)
					while ((*lst)[n] != NULL)
						n++;
				*lst = realloc(*lst, sizeof(char *) * (n + 2));
				if (*lst == NULL)
					fatal("out of memory");
				(*lst)[n    ] = val;
				(*lst)[n + 1] = NULL;
				itm->action = lst;
				break;
			}
			// Functions: just call the given function with the user
			// data and requested arguments.
			case '0': {
				((arg_func0_t)itm->action)(itm->data, arg);
				break;
			}
			case '1': {
				((arg_func1_t)itm->action)(itm->data, arg, val);
				break;
			}
		}
	}
	*argc = rem;
}

/* help:
 *   Simple callback for argument parsing function to display the help message
 *   on the standard error. Depending on the exact syntax of the switch this can
 *   display to levels of details.
 */
static
void help(void *ud, char *cmd) {
  static const char *help_msg[] = {
    " usage: lost [option]*",
    " ",
    " Global options:",
    " \t-h | --help                Display basic usage informations",
    " \t   | --Help                Display advanced usage informations",
    " \t   | --version             Display version informations",
    " \t-v | --verbose             Display more informations",
    " \t   | --nthreads     INT    Number of compute threads",
    " ",
    " Model options:",
    " \t   | --mdl-load     FILE   Model file to load",
    " \t   | --mdl-save     FILE   File to store the model",
    " \t   | --mdl-save-otf FILE   File to store the model at each iter",
    " \t   | --mdl-compact         Compact model before saving",
    "$\t   | --ftr-dump     FILE   File to dump features hash list",
    " ",
    " Data files:",
    " \t   | --train-spc    FILE   Load train spaces FSTs from file",
    " \t   | --train-ref    FILE   Load train references FSTs from file",
    " \t   | --devel-spc    FILE   Load devel FSTs from file",
    " \t   | --devel-out    FILE   Save devel results to file",
    " \t   | --test-spc     FILE   Load test FSTs from file",
    " \t   | --test-out     FILE   Save test results to file",
    " \t   | --test-fst     FILE   Save full test space to file",
    " ",
    " Features:",
    " \t   | --pattern      T:STR  Add a pattern for feature extraction",
    "$\t   | --tag-start    T:INT  Tag is introduced at iteration N",
    "$\t   | --tag-remove   T:INT  Tag is removed from iteration N",
    " \t   | --tag-rho1     T:FLT  L1 regularization for tag",
    " \t   | --tag-rho2     T:FLT  L2 regularization for tag",
    "$\t   | --tag-rho3     T:FLT  L3 regularization for tag",
    " \t   | --ref-freq            Compute frequency on ref instead of spc",
    " \t   | --min-freq     INT    Minimum frequency",
    " ",
    " Optimization:",
    "$\t   | --cache-lvl    INT    Amount of data to keep in mem (0-4)",
    " \t   | --iterations   INT    Number of optimization step to do",
    "$\t   | --rbp-stpinc   FLOAT  Step increment factor",
    "$\t   | --rbp-stpdec   FLOAT  Step decrement factor",
    "$\t   | --rbp-stpmin   FLOAT  Minimum step value",
    "$\t   | --rbp-stpmax   FLOAT  Maximum step value",
    "$",
    "$String pool:",
    "$\t   | --str-load     FILE   String pool file to preload",
    "$\t   | --str-save     FILE   Dump string pool to file",
    "$\t   | --str-all             Store all strings in the pool",
    NULL
  };
	int full = 0;
	if (cmd != NULL && !strcmp(cmd, "--Help"))
		full = 1;
	for (int i = 0; help_msg[i] != NULL; i++)
		if (full || help_msg[i][0] != '$')
			fprintf(stderr, "%s\n", help_msg[i] + 1);
	exit(EXIT_FAILURE);
	(void)ud;
}

/* version:
 * Simple callback to display version and copyright informations.
 */
static
void version(void *ud, char *cmd) {
	fprintf(stderr, "Lost v" LOST_VERSION " -- ");
	fprintf(stderr, "Copyright (c) 2013-2014  LIMSI-CNRS\n");
	exit(EXIT_FAILURE);
	(void)(cmd && ud);
}

int main(int argc, char *argv[argc]) {
	int    verbose     = 0;
	int    nthreads    = 1;
	char **str_load    = NULL,  *str_save   = NULL;
	int    str_all     = 0;
	char **mdl_inp     = NULL,  *mdl_outp   = NULL, *mdl_outp_otf = NULL;
	int    mdl_compact = 0,      ref_freq   = 0;
	char  *ftr_dump    = NULL;
	char **pos_train   = NULL, **neg_train  = NULL;
	char  *spc_test    = NULL,  *out_test   = NULL, *fst_test     = NULL;
	char  *spc_devel   = NULL,  *out_devel  = NULL;
	double rbp_stpinc  = 1.2,    rbp_stpdec = 0.5;
	double rbp_stpmin  = 1e-8,   rbp_stpmax = 50.0;
	char **tag_start   = NULL, **tag_remove = NULL;
	char **tag_rho1    = NULL, **tag_rho2   = NULL, **tag_rho3    = NULL;
	int    min_freq    = 0;
	char **pattern     = NULL;
	int    iters       = 15,     cachelvl   = 0;
	int    tick_dat    = 1000;
	if (argc <= 1)
		help(NULL, NULL);
	argc--, argv++;
	arg_t arg_def[] = {
		{'0', "-h", "--help",         (void *)&help,         NULL},
		{'0', "  ", "--Help",         (void *)&help,         NULL},
		{'0', "  ", "--version",      (void *)&version,      NULL},
		{'b', "-v", "--verbose",      (void *)&verbose,      NULL},
		{'u', "  ", "--nthreads",     (void *)&nthreads,     NULL},
		{'S', "  ", "--mdl-load",     (void *)&mdl_inp,      NULL},
		{'s', "  ", "--mdl-save",     (void *)&mdl_outp,     NULL},
		{'s', "  ", "--mdl-save-otf", (void *)&mdl_outp_otf, NULL},
		{'b', "  ", "--mdl-compact",  (void *)&mdl_compact,  NULL},
		{'s', "  ", "--ftr-dump",     (void *)&ftr_dump,     NULL},
		{'S', "  ", "--train-spc",    (void *)&pos_train,    NULL},
		{'S', "  ", "--train-ref",    (void *)&neg_train,    NULL},
		{'s', "  ", "--devel-spc",    (void *)&spc_devel,    NULL},
		{'s', "  ", "--devel-out",    (void *)&out_devel,    NULL},
		{'s', "  ", "--test-spc",     (void *)&spc_test,     NULL},
		{'s', "  ", "--test-out",     (void *)&out_test,     NULL},
		{'s', "  ", "--test-fst",     (void *)&fst_test,     NULL},
		{'S', "  ", "--pattern",      (void *)&pattern,      NULL},
		{'S', "  ", "--tag-start",    (void *)&tag_start,    NULL},
		{'S', "  ", "--tag-remove",   (void *)&tag_remove,   NULL},
		{'S', "  ", "--tag-rho1",     (void *)&tag_rho1,     NULL},
		{'S', "  ", "--tag-rho2",     (void *)&tag_rho2,     NULL},
		{'S', "  ", "--tag-rho3",     (void *)&tag_rho3,     NULL},
		{'b', "  ", "--ref-freq",     (void *)&ref_freq,     NULL},
		{'u', "  ", "--min-freq",     (void *)&min_freq,     NULL},
		{'S', "  ", "--str-load",     (void *)&str_load,     NULL},
		{'s', "  ", "--str-save",     (void *)&str_save,     NULL},
		{'b', "  ", "--str-all",      (void *)&str_all,      NULL},
		{'u', "  ", "--iterations",   (void *)&iters,        NULL},
		{'u', "  ", "--cache-lvl",    (void *)&cachelvl,     NULL},
		{'p', "  ", "--rbp-stpinc",   (void *)&rbp_stpinc,   NULL},
		{'p', "  ", "--rbp-stpdec",   (void *)&rbp_stpdec,   NULL},
		{'p', "  ", "--rbp-stpmin",   (void *)&rbp_stpmin,   NULL},
		{'p', "  ", "--rbp-stpmax",   (void *)&rbp_stpmax,   NULL},
		{0, NULL, NULL, NULL, NULL}
	};
	arg_parse(arg_def, &argc, argv);
	// System initialization:
	//   Here we do the system preparation common to all modes of operation
	//   like preparing the string pool and tuple table.
	fprintf(stderr, "* Setup the system base\n");
	fprintf(stderr, "  - Initialize string pool\n");
	ssp_t *ssp = ssp_new(str_all);
	if (str_load != NULL) {
		for (int i = 0; str_load[i] != NULL; i++) {
			fprintf(stderr, "    [str] %s\n", str_load[i]);
			if (!ssp_load(ssp, str_load[i]))
				pfatal("cannot load file %s", str_load[i]);
		}
	}
	fprintf(stderr, "  - Initialize model object\n");
	mdl_t *mdl = mdl_new(ssp);
	// Data loading:
	//   Next, we load all the datasets. The FST are stored in a simple form
	//   that do not take too much memory.
	fprintf(stderr, "* Load the data\n");
	dat_t *dat_train = NULL;
	if (pos_train != NULL) {
		int t = tick_dat;
		if (dat_train == NULL)
			dat_train = dat_new();
		for (int i = 0; pos_train[i] != NULL; i++) {
			fprintf(stderr, "    [pos] %s\n", pos_train[i]);
			if (dat_load(dat_train, pos_train[i], mdl, 1.0, t))
				pfatal("cannot load file %s", pos_train[i]);
		}
	}
	if (neg_train != NULL) {
		int t = tick_dat;
		if (dat_train == NULL)
			dat_train = dat_new();
		for (int i = 0; neg_train[i] != NULL; i++) {
			fprintf(stderr, "    [neg] %s\n", neg_train[i]);
			if (dat_load(dat_train, neg_train[i], mdl, -1.0, t))
				pfatal("cannot load file %s", neg_train[i]);
		}
	}
	dat_t *dat_devel = NULL;
	if (spc_devel != NULL) {
		int t = tick_dat;
		dat_devel = dat_new();
		fprintf(stderr, "    [spc] %s\n", spc_devel);
		if (dat_load(dat_devel, spc_devel, mdl, 0, t))
			pfatal("cannot load file %s", spc_devel);
	}
	dat_t *dat_test = NULL;
	if (spc_test != NULL) {
		int t = tick_dat;
		dat_test = dat_new();
		fprintf(stderr, "    [spc] %s\n", spc_test);
		if (dat_load(dat_test, spc_test, mdl, 0, t))
			pfatal("cannot load file %s", spc_test);
	}
	if (dat_train != NULL)
		fprintf(stderr, "        %d train FSTs\n", dat_train->nfst);
	if (dat_devel != NULL)
		fprintf(stderr, "        %d devel FSTs\n", dat_devel->nfst);
	if (dat_test != NULL)
		fprintf(stderr, "        %d test FSTs\n", dat_test->nfst);
	// The model:
	//   Now, we can go for the serious things. We need the model and a
	//   feature generator to produce some features for it.
	fprintf(stderr, "* Prepare the model\n");
	mdl->frq = min_freq;
	fprintf(stderr, "  - Initialize the feature generator\n");
	gen_t *gen = gen_new(ssp, ref_freq);
	if (pattern != NULL) {
		for (int i = 0; pattern[i] != NULL; i++)
			if (!gen_addpat(gen, pattern[i]))
				fatal("invalid pattern %s", pattern[i]);
	} else {
		fatal("no pattern specified");
	}
	fprintf(stderr, "  - Initialize the feature table\n");
	if (ftr_dump != NULL) {
		mdl->dump = fopen(ftr_dump, "w");
		nthreads = 1;
	}
	if (tag_start != NULL) {
		for (int i = 0; tag_start[i] != NULL; i++) {
			int tag, val;
			if (sscanf(tag_start[i], "%d:%d", &tag, &val) != 2)
				pfatal("bad --tag-start %s", tag_start[i]);
			mdl->stt[tag] = val;
		}
	}
	if (tag_remove != NULL) {
		for (int i = 0; tag_remove[i] != NULL; i++) {
			int tag, val;
			if (sscanf(tag_remove[i], "%d:%d", &tag, &val) != 2)
				pfatal("bad --tag-remove %s", tag_remove[i]);
			mdl->rem[tag] = val;
		}
	}
	if (mdl_inp != NULL) {
		fprintf(stderr, "  - Load previous model file\n");
		for (int i = 0; mdl_inp[i] != NULL; i++) {
			fprintf(stderr, "    [mdl] %s\n", mdl_inp[i]);
			if (!mdl_load(mdl, mdl_inp[i]))
				pfatal("cannot load file %s", mdl_inp[i]);
		}
	}
	fprintf(stderr, "  - Initialize the gradient computer\n");
	grd_t *grd = grd_new(mdl, gen, dat_train);
	grd->nth   = nthreads;
	grd->cache = cachelvl;
	fprintf(stderr, "  - Initialize the optimizer\n");
	rbp_t *rbp = rbp_new();
	rbp->stpinc = rbp_stpinc;
	rbp->stpdec = rbp_stpdec;
	rbp->stpmin = rbp_stpmin;
	rbp->stpmax = rbp_stpmax;
	if (tag_rho1 != NULL) {
		for (int i = 0; tag_rho1[i] != NULL; i++) {
			int tag; double val;
			if (sscanf(tag_rho1[i], "%d:%lf", &tag, &val) != 2) {
				if (sscanf(tag_rho1[i], "%lf", &val) != 1)
					pfatal("bad rho1 %s", tag_rho1[i]);
				tag = 0;
			}
			rbp->rho1[tag] = val;
		}
	}
	if (tag_rho2 != NULL) {
		for (int i = 0; tag_rho2[i] != NULL; i++) {
			int tag; double val;
			if (sscanf(tag_rho2[i], "%d:%lf", &tag, &val) != 2) {
				if (sscanf(tag_rho2[i], "%lf", &val) != 1)
					pfatal("bad rho2 %s", tag_rho2[i]);
				tag = 0;
			}
			rbp->rho2[tag] = val;
		}
	}
	if (tag_rho3 != NULL) {
		for (int i = 0; tag_rho3[i] != NULL; i++) {
			int tag; double val;
			if (sscanf(tag_rho3[i], "%d:%lf", &tag, &val) != 2) {
				if (sscanf(tag_rho3[i], "%lf", &val) != 1)
					pfatal("bad rho3 %s", tag_rho3[i]);
				tag = 0;
			}
			rbp->rho3[tag] = val;
		}
	}
	for (int i = 1; i < 128; i++) {
		if (rbp->rho1[i] == -1.0)
			rbp->rho1[i] = rbp->rho1[0];
		if (rbp->rho2[i] == -1.0)
			rbp->rho2[i] = rbp->rho2[0];
		if (rbp->rho3[i] == -1.0)
			rbp->rho3[i] = rbp->rho3[0];
	}
	// Optimization:
	//   Now that all is ready, if a train dataset was provided, we can
	//   start the optimization.
	if (dat_train != NULL) {
		fprintf(stderr, "* Optimize the model\n");
		for (int i = 1; i <= iters; i++) {
			fprintf(stderr, "  [%3d] Start new iteration\n", i);
			mdl->itr = i;
			fprintf(stderr, "    - Compute the gradient\n");
			double fx = grd_compute(grd);/// dat_train->nfst;
			fprintf(stderr, "    - Apply the update\n");
			rbp_step(rbp, mdl, fx);
			fprintf(stderr, "    - Compute stats\n");
			mdl_stats(mdl, verbose);
			if (dat_devel != NULL) {
				fprintf(stderr, "* Decode the devel\n");
				char buf[4096];
				sprintf(buf, out_devel, i);
				FILE *file = fopen(buf, "w");
				dec_decode(mdl, ssp, gen, dat_devel, file, 0);
				fclose(file);
			}
			if (mdl_outp_otf != NULL) {
				fprintf(stderr, "  - Save model\n");
				char buf[4096];
				sprintf(buf, mdl_outp_otf, i);
				mdl_save(mdl, buf);
			}
		}
	}
	// Decoding:
	if (dat_test != NULL) {
		if (out_test != NULL) {
			fprintf(stderr, "* Decode the test (viterbi)\n");
			FILE *file = fopen(out_test, "w");
			dec_decode(mdl, ssp, gen, dat_test, file, 0);
			fclose(file);
		}
		if (fst_test != NULL) {
			fprintf(stderr, "* Decode the test (space)\n");
			FILE *file = fopen(fst_test, "w");
			dec_decode(mdl, ssp, gen, dat_test, file, 1);
			fclose(file);
		}
	}
	// Produce output files requested by the user and cleanup things. For
	// now, we don't check if user provided name for all produced data so
	// some valuable things may be lost.
	fprintf(stderr, "* Generate outputs\n");
	if (mdl_outp != NULL) {
		if (mdl_compact) {
			fprintf(stderr, "  - Compact model\n");
			mdl_shrink(mdl);
		}
		fprintf(stderr, "  - Save model\n");
		mdl_save(mdl, mdl_outp);
	}
	if (str_save != NULL) {
		fprintf(stderr, "  - Dump string pool\n");
		ssp_save(ssp, str_save);
	}
	fprintf(stderr, "* Cleanup remaining objects\n");
	if (mdl->dump != NULL)
		fclose(mdl->dump);
	dat_free(dat_train);
	rbp_free(rbp);
	grd_free(grd);
	gen_free(gen);
	ssp_free(ssp);
	fprintf(stderr, "* Done\n");
	return EXIT_SUCCESS;
}

/*******************************************************************************
 * This is the end
 ******************************************************************************/

