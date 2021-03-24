import numpy as np

# Let's define a structured array for the halo dtype
head_struct = np.dtype([('magic', np.uint64), ('snap', np.int64),
                        ('chunk', np.int64), ('scale', np.float32),
                        ('Om', np.float32), ('Ol', np.float32),
                        ('h0', np.float32), ('bounds', np.float32, (6)),
                        ('num_halos', np.int64), ('num_particles', np.int64),
                        ('box_size', np.float32),
                        ('particle_mass', np.float32),
                        ('particle_type', np.int64),
                        ('format_revision', np.int32),
                        ('rockstar_version', np.byte, (12)),
                        ('unused', np.byte, (144))], align=True)

halo_struct = np.dtype([('id', np.int64), ('pos', np.float32, (6)),
                        ('corevel', np.float32, (3)),
                        ('bulkvel', np.float32, (3)), ('m', np.float32),
                        ('r', np.float32), ('child_r', np.float32),
                        ('vmax_r', np.float32), ('mgrav', np.float32),
                        ('vmax', np.float32), ('rvmax', np.float32),
                        ('rs', np.float32), ('klypin_rs', np.float32),
                        ('vrms', np.float32), ('J', np.float32, (3)),
                        ('energy', np.float32), ('spin', np.float32),
                        ('alt_m', np.float32, (4)), ('Xoff', np.float32),
                        ('Voff', np.float32), ('b_to_a', np.float32),
                        ('c_to_a', np.float32), ('A', np.float32, (3)),
                        ('b_to_a2', np.float32), ('c_to_a2', np.float32),
                        ('A2', np.float32, (3)), ('bullock_spin', np.float32),
                        ('kin_to_pot', np.float32), ('me_pe_b', np.float32),
                        ('m_pe_d', np.float32),
                        ('halfmass_radius', np.float32),
                        ('num_p', np.int64), ('num_child_particles', np.int64),
                        ('p_start', np.int64), ('desc', np.int64),
                        ('flags', np.int64), ('n_core', np.int64),
                        ('min_pos_err', np.float32),
                        ('min_vel_err', np.float32),
                        ('min_bulkvel_err', np.float32)], align=True)

ascii_halo_struct = np.dtype([('id', np.int64), ('internal_id', np.int64),
                              ('num_p', np.int64), ('mvir', np.float32),
                              ('mbound_vir', np.float32), ('rvir', np.float32),
                              ('vmax', np.float32), ('rvmax', np.float32),
                              ('vrms', np.float32), ('pos', np.float32, (6)),
                              ('J', np.float32, (3)), ('energy', np.float32),
                              ('spin', np.float32)])


ascii_part_struct = np.dtype([('pos', np.float32, (6)), ('particle_id', np.int64),
                              ('assigned_internal_haloid', np.int64),
                              ('internal_haloid', np.int64),
                              ('external_haloid', np.int64),
                              ('energy', np.float32)])

