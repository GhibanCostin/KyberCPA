import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import time

CHUNK_SIZE = 32
MAX_TRACES_PER_FILE = 2000
KYBER_K = 2
KYBER_Q = 3329
KYBER_QA = 26632
KYBER_Q_INV = 0x6ba8f301          # -3327
KYBER_N = 256
MONT_R_INV = 169
BARRET_CONST = 20159
KYBER_POLYBYTES = 384
KYBER_POLYVECBYTES = KYBER_K * KYBER_POLYBYTES
CIPHERTEXT_BYTES = KYBER_K * 320 + 128
PUBLIC_KEY_BYTES = KYBER_K * KYBER_POLYBYTES + 32
SECRET_KEY_BYTES = KYBER_K * KYBER_POLYBYTES + PUBLIC_KEY_BYTES + 2 * 32

zetas = [
    21932846, 3562152210, 752167598, 3417653460, 2112004045, 932791035, 2951903026, 1419184148, 1817845876, 3434425636,
    4233039261, 300609006, 975366560, 2781600929, 3889854731, 3935010590, 2197155094, 2130066389, 3598276897, 2308109491,
    2382939200, 1228239371, 1884934581, 3466679822, 1211467195, 2977706375, 3144137970, 3080919767, 945692709, 3015121229,
    345764865, 826997308, 2043625172, 2964804700, 2628071007, 4154339049, 483812778, 3288636719, 2696449880, 2122325384,
    1371447954, 411563403, 3577634219, 976656727, 2708061387, 723783916, 3181552825, 3346694253, 3617629408, 1408862808,
    519937465, 1323711759, 1474661346, 2773859924, 3580214553, 1143088323, 2221668274, 1563682897, 2417773720, 1327582262,
    2722253228, 3786641338, 1141798155, 2779020594
]

def unpack_ciphertext(ct, i):
    poly_coeffs = [0] * KYBER_N

    for j in range(KYBER_N // 4):
        poly_coeffs[4*j+0] =  (((ct[320*i+5*j+0]       | ((ct[320*i+5*j+1] & 0x03) << 8)) * KYBER_Q) + 512) >> 10
        poly_coeffs[4*j+1] = ((((ct[320*i+5*j+1] >> 2) | ((ct[320*i+5*j+2] & 0x0f) << 6)) * KYBER_Q) + 512) >> 10
        poly_coeffs[4*j+2] = ((((ct[320*i+5*j+2] >> 4) | ((ct[320*i+5*j+3] & 0x3f) << 4)) * KYBER_Q) + 512) >> 10
        poly_coeffs[4*j+3] = ((((ct[320*i+5*j+3] >> 6) | ((ct[320*i+5*j+4] & 0xff) << 2)) * KYBER_Q) + 512) >> 10

    return poly_coeffs

def get_halfword(num: int, half):
    if half == 't':
        num = num >> 16
    elif half == 'b':
        num = num & 0x0000ffff
    else:
        print('Error: invalid halfword suffix! Returning number unchanged.')

    # Make sure to keep the numbers signed
    num = num.to_bytes(3, byteorder='little', signed=True)
    num = int.from_bytes(num[:2], byteorder='little', signed=True)

    return num

def poly_reduce(poly: list):
    for i in range(KYBER_N // 2):
        tmp = poly[i] * BARRET_CONST                # smulbb \tmp, \a, \barrettconst
        tmp2 = poly[i+1] * BARRET_CONST             # smultb \tmp2, \a, \barrettconst
        tmp = tmp >> 26                             # asr \tmp, \tmp, #26
        tmp2 = tmp2 >> 26                           # asr \tmp2, \tmp2, #26
        tmp = get_halfword(tmp, 'b') * KYBER_Q      # smulbb \tmp, \tmp, \q
        tmp2 = get_halfword(tmp2, 'b') * KYBER_Q    # smulbb \tmp2, \tmp2, \q
        poly[i] -= tmp                              # pkhbt \tmp, \tmp, \tmp2, lsl#16
        poly[i+1] -= tmp2                           # usub16 \a, \a, \tmp

    return poly

def poly_tobytes(poly: list):
    coeffs = poly_reduce(poly)
    poly_bytes = bytearray(3 * (KYBER_N // 2))

    for i in range(KYBER_N // 2):
        t0 = coeffs[2 * i]
        t1 = coeffs[2 * i + 1]
        poly_bytes[3 * i] = t0 & 0xff
        poly_bytes[3 * i + 1] = (t0 >> 8) | ((t1 & 0xf) << 4)
        poly_bytes[3 * i + 2] = (t1 >> 4) & 0xff

    return poly_bytes

def poly_frombytes(poly_bytes: bytearray):
    coeffs = [0] * KYBER_N

    for i in range(KYBER_N // 2):
        coeffs[2 * i] = poly_bytes[3 * i] | ((poly_bytes[3 * i + 1] & 0x0f) << 8)
        coeffs[2 * i + 1] = (poly_bytes[3 * i + 1] >> 4) | ((poly_bytes[3 * i + 2] & 0xff) << 4)

    return coeffs

def polyvec_tobytes(polyvec: list):
    polyvec_bytes = b''

    for poly in polyvec:
        polyvec_bytes += poly_tobytes(poly)

    return polyvec_bytes

def polyvec_frombytes(polyvec_bytes: bytearray):
    vec = []

    for i in range(KYBER_K):
        start = i * KYBER_POLYBYTES
        vec.append(poly_frombytes(polyvec_bytes[start:]))

    return vec

def load_traces(traces_file, ciphertexts_file, secret_key_file, ntraces) -> "tuple[list[list], tuple[list, list], list[int]]":
    combined_traces = np.load(traces_file)
    traces = [trace.tolist() for trace in combined_traces]

    with open(secret_key_file, 'r') as sk_file:
        secret_key = eval(sk_file.readline().replace('bytearray', '').replace('(', '[').replace(')', ']'))[0]
        secret_key += eval(sk_file.readline().replace('bytearray', '').replace('(', '[').replace(')', ']'))[0]

    with open(ciphertexts_file, 'r') as f:
        ntt_cts, mul_res = ''.join(f.readlines()).replace('ntt_cts: ', '').split('\nmul_res:')
        ntt_cts = ntt_cts.replace('\n\t', ',')
        mul_res = mul_res.replace('\n\t', ',')
        ntt_ciphertexts = eval('[' + ntt_cts + ']')
        ntt_mul_results = eval('[' + mul_res + ']')

    sk_polyvec = polyvec_frombytes(secret_key[:SECRET_KEY_BYTES])
    sk_first_poly = sk_polyvec[0]

    return traces[:ntraces], (ntt_ciphertexts, ntt_mul_results), sk_first_poly

# Load and visualize the power traces
ntraces = 2000
traces_file = 'plain_traces.npy'
ciphertexts_file = 'ntt_target_results.txt'
secret_key_file = 'secret_key.txt'
traces, ciphertexts, known_s = load_traces(traces_file, ciphertexts_file, secret_key_file, ntraces)
plt.plot(traces[0])
# -----------------------------------

def hamming_weight(num):
    num -= (num >> 1) & 0x5555
    num = (num & 0x3333) + ((num >> 2) & 0x3333)
    num = (num + (num >> 4)) & 0x0F0F
    num += num >> 8

    return num & 0x1F

def compute_r0(u0, u1, s0, s1, zeta):
    tmp = (zeta * s1) >> 16                             # smulwt(zeta, poly1)           result on 48 bits, return the most significant 32 (thus >> 16)
    tmp = get_halfword(tmp, 'b') * KYBER_Q + KYBER_QA   # smlabt(tmp, q, qa)
    tmp = u1 * get_halfword(tmp, 't')                   # smultt(poly0, tmp)
    tmp = u0 * s0 + tmp                                 # smlabb(poly0, poly1, tmp)
    tmp = (tmp * KYBER_Q_INV) & 0xffffffff              # mul(a, q_inv)
    tmp = get_halfword(tmp, 't') * KYBER_Q + KYBER_QA   # smlatt(tmp, q, qa)
    r0 = get_halfword(tmp, 't')                         # result in high half

    return r0

def cpa_attack_phase1(traces, ciphertexts, known_s0, known_s1, coeff_ind=0):
    corrs = []
    H = []
    zeta = zetas[coeff_ind // 4] * (-1 if (coeff_ind // 2) % 2 == 1 else 1)

    for i in range(ntraces):
        u0: int = ciphertexts[0][i][coeff_ind]
        u1: int = ciphertexts[0][i][coeff_ind + 1]
        r0 = compute_r0(u0, u1, known_s0, known_s1, zeta)
        H.append(hamming_weight(r0))

    trace_len = min(len(trace) for trace in traces)

    for j in range(trace_len):
        T = [trace[j] for trace in traces]
        corrs.append(np.corrcoef(T, H)[0, 1])

    return np.nanargmax(np.abs(corrs)), corrs

# Find where the target function is performed
known_s0 = known_s[0]
known_s1 = known_s[1]

print(known_s0, known_s1)

ind_max_corr, corrs = cpa_attack_phase1(traces, ciphertexts, known_s0, known_s1, 0)
print(f'found target function sample point: {ind_max_corr}')

fig = plt.figure(figsize=(20, 10))
ax = fig.gca()
ax.set_xlabel('Samples')
ax.set_ylabel('Correct key correlation')
ax.set_title('Correlation between the first correct secret key coefficients and trace samples')
ax.plot(range(len(corrs)), corrs)
# ------------------------------------------

def cpa_attack_phase2(traces, ciphertexts, known_s0, known_sample_point, coeff_ind=0):
    corrs = []
    T = [trace[known_sample_point] for trace in traces]
    zeta = zetas[coeff_ind // 4] * (-1 if (coeff_ind // 2) % 2 == 1 else 1)

    for s1 in range(1, KYBER_Q):
        H = []

        tmp1 = (zeta * s1) >> 16
        tmp1 = get_halfword(tmp1, 'b') * KYBER_Q + KYBER_QA

    for i in range(ntraces):
        u0: int = ciphertexts[0][i][coeff_ind]
        u1: int = ciphertexts[0][i][coeff_ind + 1]

        tmp = u1 * get_halfword(tmp1, 't')
        tmp = u0 * known_s0 + tmp
        tmp = (tmp * KYBER_Q_INV) & 0xffffffff
        tmp = get_halfword(tmp, 't') * KYBER_Q + KYBER_QA
        r0 = get_halfword(tmp, 't')
        H.append(hamming_weight(r0))

    corrs.append(np.corrcoef(T, H)[0, 1])

    s1 = np.nanargmax(np.abs(corrs))
    max_corr = corrs[s1]

    return s1, max_corr, corrs

# Find the second secret coefficient having fixed the first coefficient (intermediary step)
s1, max_corr, corrs = cpa_attack_phase2(traces, ciphertexts, known_s0, ind_max_corr, 0)
print(f'found s1: {s1} with correlation: {max_corr}')

fig = plt.figure(figsize=(20, 10))
ax = fig.gca()
ax.set_xlabel('Samples')
ax.set_ylabel('Correct key correlation')
ax.set_title('Correlation between the first pair of secret key coefficients and trace samples')
ax.plot(range(len(corrs)), corrs)
# -----------------------------

def cpa_worker(s0, s1, is_correct_key, coeff_ind, ntraces, ciphertexts, T):
    H = []
    checkpoints = [20, 50, 80]
    checkpoint_corrs = []
    plottable = True

    tmp1 = ((zetas[coeff_ind]) * s1) >> 16
    tmp1 = get_halfword(tmp1, 'b') * KYBER_Q + KYBER_QA

    for i in range(ntraces):
        u0: int = ciphertexts[0][i][coeff_ind]
        u1: int = ciphertexts[0][i][coeff_ind + 1]

        tmp = u1 * get_halfword(tmp1, 't')
        tmp = u0 * s0 + tmp
        tmp = (tmp * KYBER_Q_INV) & 0xffffffff
        tmp = get_halfword(tmp, 't') * KYBER_Q + KYBER_QA
        r0 = get_halfword(tmp, 't')

        H.append(hamming_weight(r0))

        if i in checkpoints and plottable:
            corr = np.abs(np.corrcoef(T[:i], H[:i])[0, 1])
            checkpoint_corrs.append(corr)

            if i == checkpoints[1] and corr < 0.5 and not is_correct_key:
                plottable = False
                checkpoint_corrs = []


    full_corr = np.corrcoef(T, H)[0, 1]
    plottable = is_correct_key or plottable

    if plottable:
        checkpoint_corrs.append(abs(full_corr))

    print(s0, s1)
    return (s0, s1), (full_corr, (checkpoint_corrs, is_correct_key))

# Parallelized version of the full attack
def cpa_attack_full(traces, ciphertexts, coeff_ind, known_sample_point, known_s0=None, known_s1=None):
    ntraces = len(traces)

    try:
        T = [trace[known_sample_point] for trace in traces]
        corrs = [0] * (KYBER_Q * KYBER_Q)

        s = time.time()

        def s0_s1_generator():
            for s0 in range(1, KYBER_Q):
                for s1 in range(1, KYBER_Q):
                    is_correct_key = (s0 == known_s0 and s1 == known_s1) or (s0 == KYBER_Q - known_s0 and s1 == KYBER_Q - known_s1)
                    yield (s0, s1, is_correct_key, coeff_ind, ntraces, ciphertexts, T)

        num_of_cores = mp.cpu_count()

        print(f'Running the attack on {num_of_cores} cores')

        with mp.Pool(processes=num_of_cores) as pool:
            corrs = pool.map(cpa_worker, s0_s1_generator())

        e = time.time()

        print(f'Attack took {e - s}s')
    except KeyboardInterrupt:
        print('Stopping the attack')

    return corrs

# Naive implementation
def cpa_attack_full2(traces, ciphertexts, coeff_ind, known_sample_point):
    ntraces = len(traces)
    T = [trace[known_sample_point] for trace in traces]
    H = []
    corrs = []

    for s0 in range(KYBER_Q):
        for s1 in range(KYBER_Q):
            H = []
            tmp1 = ((zetas[coeff_ind]) * s1) >> 16
            tmp1 = get_halfword(tmp1, 'b') * KYBER_Q + KYBER_QA

            for i in range(ntraces):
                u0: int = ciphertexts[0][i][coeff_ind]
                u1: int = ciphertexts[0][i][coeff_ind + 1]

                tmp = u1 * get_halfword(tmp1, 't')
                tmp = u0 * s0 + tmp
                tmp = (tmp * KYBER_Q_INV) & 0xffffffff
                tmp = get_halfword(tmp, 't') * KYBER_Q + KYBER_QA
                r0 = get_halfword(tmp, 't')

                H.append(hamming_weight(r0))
            
            corrs.append(np.corrcoef(T, H)[0, 1])

    corrs = np.abs(corrs)
    s01 = np.nanargmax(corrs)
    s0, s1 = 3100 + s01 // (KYBER_Q - 1), (s01 % (KYBER_Q - 1)) + 1
    max_corr = corrs[s01]

    return (s0, s1), max_corr, corrs

# Perform the full attack, finding both secret coefficients
ntraces = 80
subtraces = traces[:ntraces]
(s0, s1), max_corr, corrs = cpa_attack_full2(subtraces, ciphertexts, 0, ind_max_corr)

print(f'found (s0, s1) = {(s0, s1)} with corr = {max_corr}')

fig = plt.figure(figsize=(20, 10))
ax = fig.gca()
ax.set_xlabel('Samples')
ax.set_ylabel('Correct key correlation')
ax.set_title('Correlation between the first pair of secret key coefficients and trace samples')
ax.plot(range(len(corrs)), corrs)
# ----------------------------------------------------------

# Determine the minimum number of traces - empiric version
checkpoints = [20, 50, 80, 100, 200, 300, 500, 1000]

fig = plt.figure(figsize=(20, 10))
ax = fig.gca()
ax.set_xlabel('Number of traces')
ax.set_ylabel('Correlation')
ax.set_title('Evolution of correlation with the number of traces')
ax.set_xticks(checkpoints)

plot_corrs = []

for ntraces in checkpoints:
  subtraces = traces[:ntraces]
  subct = ciphertexts[:ntraces]
  s1, max_corr, corrs = cpa_attack_phase2(subtraces, subct, known_s0, ind_max_corr, 0)
  plot_corrs.append(np.abs(corrs[:20] + [max_corr]))

ax.plot(checkpoints, plot_corrs)
# --------------------------------------------------------