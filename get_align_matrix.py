def align_stack(stack_img, auto_save=True):
    sr = StackReg(StackReg.TRANSLATION)
    tmats = sr.register_stack(stack_img, reference='previous')
    out1 = sr.transform_stack(stack_img)

    a = tmats[:, 0, 2]
    b = tmats[:, 1, 2]
    trans_matrix = np.column_stack([a, b])

    return np.float32(out1), trans_matrix


def get_aligned_tiff(files=glob('xrf3/*/detsum_Au_L_norm.tiff'), dest='XRF_Data/', flip_scan=107314, elem='Au_L'):
    # files = glob('xrf3/*/detsum_Au_L_norm.tiff')
    # dest = 'xrf3/Au_Files3/'
    im = np.array(tf.imread(files[0])) * 0

    # files[0][23:-22]

    for n, i in enumerate(sorted(files)):
        img = np.array(tf.imread(i))
        sd = int(i[0].split('/')[1].split('_')[-1])
        if sd > flip_scan:
            logger.info(f'flipped: {sd}')
            img = np.fliplr(img)
        im = np.dstack((im, img))
    img_stack = np.transpose(im, (2, 1, 0))
    img_stack = np.fliplr(np.rot90(img_stack[1:, :, :], axes=(1, 2)))

    tf.imsave(dest + elem + '_stack_raw.tiff', data=img_stack, imagej=True)

    al_stack, tfm_matrix = align_stack(img_stack, auto_save=True)
    logger.info("'Stack aligned'.")
    # print('stack aligned')
    tf.imsave(dest + elem + '_stack_aligned.tiff', data=al_stack)
    logger.info('stack saved')
    np.savetxt(dest + elem + 'Transformation_Matrix.txt', tfm_matrix)
    imageio.mimsave(dest + elem + '_stack_aligned.gif', al_stack)
    logger.info('gif saved')
    logger.info('process completed')

    # tf.imsave(dest+i[24:-22]+'Au_L.tiff', data=img)
