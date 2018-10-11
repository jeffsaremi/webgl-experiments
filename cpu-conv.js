function computeSizeAndPad(
    inSize, stride, kernel, pads, padHeadIndex, padTailIndex, autoPad) {
  if (autoPad && autoPad !== 'NOTSET') {
    switch (autoPad) {
      case 'VALID':
        pads[padHeadIndex] = 0;
        pads[padTailIndex] = 0;
        return Math.floor(((inSize - kernel) / stride) + 1);
      case 'SAME_LOWER':
        const legacyTargetSize1 = (inSize + stride - 1) / stride;
        const padNeeded1 = (legacyTargetSize1 - 1) * stride + kernel - inSize;
        pads[padHeadIndex] = Math.floor((padNeeded1 + 1) / 2);
        pads[padTailIndex] = padNeeded1 - pads[padHeadIndex];
        return Math.floor(((inSize + padNeeded1 - kernel) / stride) + 1);
      case 'SAME_UPPER':
        const legacyTargetSize = (inSize + stride - 1) / stride;
        const padNeeded = (legacyTargetSize - 1) * stride + kernel - inSize;
        pads[padHeadIndex] = Math.floor(padNeeded / 2);
        pads[padTailIndex] = padNeeded - pads[padHeadIndex];
        return Math.floor(((inSize + padNeeded - kernel) / stride) + 1);
      default:
        throw `Unsupported AutoPad type`;
    }
  } else {
    return Math.floor(((inSize + pads[padHeadIndex] + pads[padTailIndex] - kernel) / stride) + 1);
  }
}

function inferOutputSize(
    isGlobalOperator, inputDims, outputDims, strides, kernelShape, pads, autoPad) {
  if (isGlobalOperator) {
    for (let dim = 0; dim < inputDims.length - 2; dim++) {
      outputDims.push(1);
    }
  } else {
    for (let dim = 0; dim < inputDims.length - 2; dim++) {
      outputDims.push(computeSizeAndPad(
          inputDims[dim + 2], strides[dim], kernelShape[dim], pads, dim, dim + inputDims.length - 2, autoPad));
    }
  }
}

function calcOutputShape(inputShape, kernelShape, autoPad, dilations, pads, strides) {
  const spatialRank = inputShape.length - 2;
  const inputSpatialShape = inputShape.slice(2);
  // make copy of pads as it might be mutated based on 'autoPad' attribute
  const adjustPads = pads.slice(0);
  inferOutputSize(false, inputShape, [], strides, kernelShape, adjustPads, autoPad);
  const inputSpatialShapeWithPad = inputSpatialShape.map((v, i) => v + adjustPads[i] + adjustPads[i + spatialRank]);
  const batchSize = inputShape[0];
  const outChannels = kernelShape[0];
  
  const kernelSpatialShape = kernelShape.slice(2);
  const dilatedKernelShape = kernelSpatialShape.map((v, i) => v + (v - 1) * (dilations[i] - 1));
  const outputSpatialShape =
      inputSpatialShapeWithPad.map((v, i) => Math.floor((v - dilatedKernelShape[i] + strides[i]) / strides[i]));
  return [batchSize, outChannels].concat(...outputSpatialShape);
}
function cpuConv(
    x, x_dims, w, kernelShape, b, autoPad, dilations, group,
    pads, strides) {
  let ndx = ndarray(x, x_dims.slice(0)).transpose(0, 2, 3, 1);
  const ndk = ndarray(w, kernelShape.slice(0)).transpose(2, 3, 1, 0);

  // adjust pads based on 'autoPad' attribute if needed
  inferOutputSize(false, x_dims, [], strides, kernelShape, pads, autoPad);

  // padding if needed
  const localPads = [[0, 0], [pads[0], pads[2]], [pads[1], pads[3]], [0, 0]];
  const padTotal = localPads.reduce((s, p) => s + p[0] + p[1], 0);
  if (padTotal !== 0) {
    const shape = ndx.shape;
    const newShape = shape.map((len, index) => len + localPads[index][0] + localPads[index][1]);
    const newSize = newShape.reduce((m, v) => m * v, 1);
    const ndp = ndarray(new Float32Array(newSize), newShape);
    const hiPoint = localPads.map((pair, index) => newShape[index] - pair[1]);
    const loPoint = localPads.map(pair => pair[0]);
    const originalSlice = ndp.hi(...hiPoint).lo(...loPoint);
    nd_ops.assign(originalSlice, ndx);
    ndx = ndp;
  }

  const [batchSize, xRows, xCols, xChannels] = ndx.shape;
  const [wRows, wCols, xChannelsInW, yChannels] = ndk.shape;

  if (xChannelsInW !== xChannels) {
    throw 'Input Channel not matching in Input and Kernel!';
  }

  // calulate the patch view in srouce image's size after dilations
  const pvRows = wRows + (wRows - 1) * (dilations[0] - 1);
  const pvCols = wCols + (wCols - 1) * (dilations[1] - 1);

  const yRows = Math.floor((xRows - pvRows + strides[0]) / strides[0]);
  const yCols = Math.floor((xCols - pvCols + strides[1]) / strides[1]);

  const ySize = batchSize * yRows * yCols * yChannels;
  const patchSize = wRows * wCols * xChannels;

  const ndf = ndarray(new Float64Array(ndk.size), [patchSize, yChannels]);
  const patch = ndarray(new Float64Array(patchSize), [wRows, wCols, xChannels]);
  for (let yChannel = 0; yChannel < yChannels; ++yChannel) {
    nd_ops.assign(patch, ndk.pick(null, null, null, yChannel));
    const reshapedPatch = ndarray(patch.data, [patchSize]);
    nd_ops.assign(ndf.pick(null, yChannel), reshapedPatch);
  }

  const yArray = new Float64Array(ySize);
  const pixelVec = ndarray(new Float64Array(yChannels), [1, yChannels]);
  let offset = 0;
  for (let b = 0; b < batchSize; ++b) {
    const image = ndx.pick(b, null, null, null);
    for (let yRow = 0; yRow < yRows; ++yRow) {
      const xRowStart = yRow * strides[0];
      for (let yCol = 0; yCol < yCols; ++yCol) {
        const xColStart = yCol * strides[1];

        const patchView = image.hi(xRowStart + pvRows, xColStart + pvCols, xChannels)
                              .lo(xRowStart, xColStart, 0)
                              .step(dilations[0], dilations[1], 1);
        nd_ops.assign(patch, patchView);
        const pvVec = ndarray(patch.data, [1, patchSize]);
        nd_gemm(pixelVec, pvVec, ndf);
        yArray.set(pixelVec.data, offset);
        offset += yChannels;
      }
    }
  }
  const ndy = ndarray(yArray, [batchSize, yRows, yCols, yChannels]);
  const ndyTransed = ndarray(new Float32Array(ySize), [batchSize, yChannels, yRows, yCols]);
  nd_ops.assign(ndyTransed, ndy.transpose(0, 3, 1, 2));
  return ndyTransed.data;
}
