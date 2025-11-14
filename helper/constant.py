
# NSL-KDD dataset - All categorical features (for full 42-feature dataset)
# Causal features (10 from CRFS): service, dst_bytes, src_bytes, diff_srv_rate, dst_host_srv_count,
#                                  dst_host_same_srv_rate, dst_host_diff_srv_rate, dst_host_serror_rate, count, flag
# All categorical features in NSL-KDD: protocol_type, service, flag, label
NSL_KDD_CATEGORY = ['protocol_type', 'service', 'flag', 'label']