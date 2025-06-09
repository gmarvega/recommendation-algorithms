import numpy as np
from scipy.sparse import csr_matrix
from operator import itemgetter

def data_masks(all_sessions, n_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j]) #count the unique items in a session，delete the repeat items, ranking by item_id
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i]-1)
            data.append(1)
    # indptr:sum of the session length; indices:item_id - 1
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))
    # 10000 * 6558 #sessions * #items H
    return matrix

def data_easy_masks(data_l, n_row, n_col):
    print(f"[DEBUG data_easy_masks] Recibido: n_row={n_row}, n_col={n_col}, len(data_l[0])={len(data_l[0]) if data_l and len(data_l) > 0 else 'N/A'}, len(data_l[1])={len(data_l[1]) if data_l and len(data_l) > 1 else 'N/A'}, len(data_l[2])={len(data_l[2]) if data_l and len(data_l) > 2 else 'N/A'} (indptr)") # LOG
    data, indices, indptr  = data_l[0], data_l[1], data_l[2]
    print(f"[DEBUG data_easy_masks] Creando csr_matrix con shape=({n_row}, {n_col}), len(indptr)={len(indptr)}") # LOG
    matrix = csr_matrix((data, indices, indptr), shape=(n_row, n_col))
    # 10000 * 6558 #sessions * #items H
    return matrix

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

class Data():
    def __init__(self, data, shuffle=False, n_node=None, n_price=None, n_category=None):
        print(f"[DEBUG Data.__init__] n_node={n_node}, n_price={n_price}, n_category={n_category}") # LOG
        # Padding para self.raw (data[0])
        item_sequences = data[0]
        max_len_items = 0 # Inicializar max_len_items aquí
        if not item_sequences:
            self.raw = np.array([])
        else:
            # Encontrar la longitud máxima de las secuencias de ítems
            # max_len_items ya está inicializada
            for seq in item_sequences:
                if len(seq) > max_len_items:
                    max_len_items = len(seq)
            
            # Aplicar padding a las secuencias de ítems
            padded_item_sequences = []
            for seq in item_sequences:
                padded_seq = seq + [0] * (max_len_items - len(seq))
                padded_item_sequences.append(padded_seq)
            self.raw = np.asarray(padded_item_sequences) # sessions, item_seq

        # Padding para self.price_raw (data[1]) - Asumiendo que también podría necesitarlo y debe tener la misma forma
        price_sequences = data[1]
        if not price_sequences:
            self.price_raw = np.array([])
        else:
            # Usar la misma max_len_items para consistencia si las longitudes deben coincidir
            # o calcular una max_len_prices separada si pueden ser diferentes
            # Por ahora, asumiremos que deben coincidir con las secuencias de ítems
            max_len_prices = max_len_items # O calcular len(p) for p in price_sequences

            padded_price_sequences = []
            for seq in price_sequences:
                # Asegurarse de que seq no sea None o un tipo inesperado si data[1] puede ser heterogéneo
                if isinstance(seq, list):
                     padded_seq = seq + [0] * (max_len_prices - len(seq))
                else:
                    # Manejar el caso donde un elemento de price_sequences no es una lista
                    # Esto podría ser un error en los datos o requerir una lógica diferente
                    # Por ahora, creamos una lista de ceros con la longitud esperada
                    print(f"WARN: Elemento inesperado en price_sequences: {seq}. Usando padding con ceros.")
                    padded_seq = [0] * max_len_prices
                padded_price_sequences.append(padded_seq)
            self.price_raw = np.asarray(padded_price_sequences) # price_seq

        print(f"[DEBUG Data.__init__] Llamando a data_easy_masks para H_T con data[2], n_row=len(data[0])={len(data[0])}, n_col=n_node={n_node}") # LOG
        print(f"[DEBUG Data.__init__]   data[2] lengths: data={len(data[2][0]) if data[2] and len(data[2]) > 0 else 'N/A'}, indices={len(data[2][1]) if data[2] and len(data[2]) > 1 else 'N/A'}, indptr={len(data[2][2]) if data[2] and len(data[2]) > 2 else 'N/A'}") # LOG
        H_T = data_easy_masks(data[2], len(data[0]), n_node)  # 10000 * 6558 #sessions * #items
        
        # Log para la división por cero
        sum_axis1_H_T = H_T.sum(axis=1)
        print(f"[DEBUG Data.__init__] H_T.sum(axis=1) para BH_T: {sum_axis1_H_T}") # LOG
        if np.any(sum_axis1_H_T == 0):
            print(f"[WARN Data.__init__] H_T.sum(axis=1) contiene ceros. Índices: {np.where(sum_axis1_H_T == 0)[0]}") # LOG
        
        # Evitar división por cero directamente aquí para el log, el error original ocurrirá si no se corrige
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_sum_H_T = 1.0 / sum_axis1_H_T
            inv_sum_H_T[~np.isfinite(inv_sum_H_T)] = 0 # Reemplazar inf con 0 para la multiplicación, aunque esto altera el cálculo original
        
        BH_T = H_T.T.multiply(inv_sum_H_T.reshape(1, -1))
        BH_T = BH_T.T
        H = H_T.T
        sum_axis1_H = H.sum(axis=1)
        print(f"[DEBUG Data.__init__] H.sum(axis=1) para DH: {sum_axis1_H}") # LOG
        if np.any(sum_axis1_H == 0):
            print(f"[WARN Data.__init__] H.sum(axis=1) contiene ceros. Índices: {np.where(sum_axis1_H == 0)[0]}") # LOG
        
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_sum_H = 1.0 / sum_axis1_H
            inv_sum_H[~np.isfinite(inv_sum_H)] = 0
            
        DH = H.T.multiply(inv_sum_H.reshape(1, -1))
        DH = DH.T
        DHBH_T = np.dot(DH, BH_T)

        print(f"[DEBUG Data.__init__] Llamando a data_easy_masks para H_pv con data[4], n_row=n_price={n_price}, n_col=n_node={n_node}") # LOG
        print(f"[DEBUG Data.__init__]   data[4] lengths: data={len(data[4][0]) if data[4] and len(data[4]) > 0 else 'N/A'}, indices={len(data[4][1]) if data[4] and len(data[4]) > 1 else 'N/A'}, indptr={len(data[4][2]) if data[4] and len(data[4]) > 2 else 'N/A'}") # LOG
        H_pv = data_easy_masks(data[4], n_price, n_node)
        BH_pv = H_pv

        BH_vp = H_pv.T

        print(f"[DEBUG Data.__init__] Llamando a data_easy_masks para H_pc con data[5], n_row=n_price={n_price}, n_col=n_category={n_category}") # LOG
        print(f"[DEBUG Data.__init__]   data[5] lengths: data={len(data[5][0]) if data[5] and len(data[5]) > 0 else 'N/A'}, indices={len(data[5][1]) if data[5] and len(data[5]) > 1 else 'N/A'}, indptr={len(data[5][2]) if data[5] and len(data[5]) > 2 else 'N/A'}") # LOG
        H_pc = data_easy_masks(data[5], n_price, n_category)
        BH_pc = H_pc

        BH_cp = H_pc.T

        print(f"[DEBUG Data.__init__] Llamando a data_easy_masks para H_cv con data[6], n_row=n_category={n_category}, n_col=n_node={n_node}") # LOG
        print(f"[DEBUG Data.__init__]   data[6] lengths: data={len(data[6][0]) if data[6] and len(data[6]) > 0 else 'N/A'}, indices={len(data[6][1]) if data[6] and len(data[6]) > 1 else 'N/A'}, indptr={len(data[6][2]) if data[6] and len(data[6]) > 2 else 'N/A'}") # LOG
        H_cv = data_easy_masks(data[6], n_category, n_node)
        BH_cv = H_cv
        
        BH_vc = H_cv.T


        self.adjacency = DHBH_T.tocoo()

        self.adjacency_pv = BH_pv.tocoo()
        self.adjacency_vp = BH_vp.tocoo()
        self.adjacency_pc = BH_pc.tocoo()
        self.adjacency_cp = BH_cp.tocoo()
        self.adjacency_cv = BH_cv.tocoo()
        self.adjacency_vc = BH_vc.tocoo()

        self.n_node = n_node
        self.n_price = n_price
        self.n_category = n_category
        self.targets = np.asarray(data[7])
        self.length = len(self.raw)
        self.shuffle = shuffle

    def get_overlap(self, sessions):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)
            for j in range(i+1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap))/float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0]*len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0/degree)
        return matrix, degree

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            # random session item_seq & price_seq
            self.raw = self.raw[shuffled_arg]
            self.price_raw = self.price_raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    def get_slice(self, index):
        # items, num_node, price_seqs = [], [], [] # num_node no se usa de la misma forma
        items, price_seqs = [], []
        inp = self.raw[index] # inp ya es un array de sesiones rellenadas
        inp_price = self.price_raw[index] # inp_price ya es un array de secuencias de precios rellenadas

        session_len_list = [] # Renombrado para evitar confusión con la variable session_len retornada
        mask_list = [] # Renombrado
        reversed_sess_item_list = [] # Renombrado

        # Ya no necesitamos calcular max_n_node de la misma manera,
        # la longitud de la sesión ya es la longitud máxima (rellenada)
        # num_node (longitudes originales) todavía se calcula para session_len

        num_node_orig = [] # Para almacenar las longitudes originales de las sesiones en el batch
        for session_row in inp: # session_row es una fila de self.raw (una sesión ya rellenada)
            # Contar elementos no cero para obtener la longitud original
            # Asegurarse de que session_row es un array NumPy para np.nonzero
            if not isinstance(session_row, np.ndarray):
                session_row = np.array(session_row)
            nonzero_elems_indices = np.nonzero(session_row)[0]
            original_length = len(nonzero_elems_indices)
            num_node_orig.append(original_length)

        for i in range(len(inp)):
            session = inp[i] # session ya está rellenada
            price = inp_price[i] # price ya está rellenada

            # Asegurarse de que session y price son arrays NumPy para operaciones
            if not isinstance(session, np.ndarray):
                session = np.array(session)
            if not isinstance(price, np.ndarray):
                price = np.array(price)

            nonzero_elems_indices = np.nonzero(session)[0]
            original_length = len(nonzero_elems_indices)
            
            session_len_list.append([original_length])
            items.append(list(session)) # Usar la sesión rellenada directamente
            price_seqs.append(list(price)) # Usar la secuencia de precios rellenada directamente

            current_mask = [1] * original_length + [0] * (len(session) - original_length)
            mask_list.append(current_mask)

            # Para reversed_sess_item, invertir solo los elementos originales y luego rellenar
            original_items = session[nonzero_elems_indices]
            reversed_original_items = list(original_items[::-1])
            reversed_padded_session = reversed_original_items + [0] * (len(session) - original_length)
            reversed_sess_item_list.append(reversed_padded_session)

        return self.targets[index]-1, session_len_list, items, reversed_sess_item_list, mask_list, price_seqs


