const testDataSet = getTestData();

function createRandomArray(size) {
    return new Float32Array(Array.from({length: size}, (v,k) => k % 10));
    //return new Float32Array(Array.from({length: size}, () => Math.floor(Math.random() * 10)));
}

function getTestData() {
    return [
    //   {
    //     inputShape: [1, 3, 7, 7],
    //     kernelShape: [8, 3, 3, 3],
    //     bias: false,
    //     outputShape: [1, 8, 10, 10],
    //     paddings: [1, 1, 1, 1],
    //     dilations: [1, 1],
    //     strides: [1, 1],
    //     group: 1
    //    }
        {
            inputShape: [1,3,224,224],
            kernelShape: [64,3,7,7],
            bias: false,
            outputShape: [1,64,112,112],
            paddings: [3,3,3,3],
            dilations: [1,1],
            strides: [2,2],
            group: 1
        },
        {
            inputShape: [1,64,56,56],
            kernelShape: [64,64,1,1],
            bias: false,
            outputShape: [1,64,56,56],
            paddings: [0,0,0,0],
            dilations: [1,1],
            strides: [1,1],
            group: 1
        },
        {
            inputShape: [1, 64, 56, 56],
            kernelShape: [256, 64, 1, 1],
            bias: false,
            outputShape: [1, 256, 56, 56],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 64, 56, 56],
            kernelShape: [64, 64, 3, 3],
            bias: false,
            outputShape: [1, 64, 56, 56],
            paddings: [1, 1, 1, 1],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 64, 56, 56],
            kernelShape: [256, 64, 1, 1],
            bias: false,
            outputShape: [1, 256, 56, 56],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 256, 56, 56],
            kernelShape: [64, 256, 1, 1],
            bias: false,
            outputShape: [1, 64, 56, 56],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 64, 56, 56],
            kernelShape: [64, 64, 3, 3],
            bias: false,
            outputShape: [1, 64, 56, 56],
            paddings: [1, 1, 1, 1],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 64, 56, 56],
            kernelShape: [256, 64, 1, 1],
            bias: false,
            outputShape: [1, 256, 56, 56],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 256, 56, 56],
            kernelShape: [64, 256, 1, 1],
            bias: false,
            outputShape: [1, 64, 56, 56],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 64, 56, 56],
            kernelShape: [64, 64, 3, 3],
            bias: false,
            outputShape: [1, 64, 56, 56],
            paddings: [1, 1, 1, 1],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 64, 56, 56],
            kernelShape: [256, 64, 1, 1],
            bias: false,
            outputShape: [1, 256, 56, 56],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 256, 56, 56],
            kernelShape: [128, 256, 1, 1],
            bias: false,
            outputShape: [1, 128, 56, 56],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 256, 56, 56],
            kernelShape: [512, 256, 1, 1],
            bias: false,
            outputShape: [1, 512, 28, 28],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [2, 2],
            group: 1
        },
        {
            inputShape: [1, 128, 56, 56],
            kernelShape: [128, 128, 3, 3],
            bias: false,
            outputShape: [1, 128, 28, 28],
            paddings: [1, 1, 1, 1],
            dilations: [1, 1],
            strides: [2, 2],
            group: 1
        },
        {
            inputShape: [1, 128, 28, 28],
            kernelShape: [512, 128, 1, 1],
            bias: false,
            outputShape: [1, 512, 28, 28],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 512, 28, 28],
            kernelShape: [128, 512, 1, 1],
            bias: false,
            outputShape: [1, 128, 28, 28],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 128, 28, 28],
            kernelShape: [128, 128, 3, 3],
            bias: false,
            outputShape: [1, 128, 28, 28],
            paddings: [1, 1, 1, 1],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 128, 28, 28],
            kernelShape: [512, 128, 1, 1],
            bias: false,
            outputShape: [1, 512, 28, 28],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 512, 28, 28],
            kernelShape: [128, 512, 1, 1],
            bias: false,
            outputShape: [1, 128, 28, 28],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 128, 28, 28],
            kernelShape: [128, 128, 3, 3],
            bias: false,
            outputShape: [1, 128, 28, 28],
            paddings: [1, 1, 1, 1],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 128, 28, 28],
            kernelShape: [512, 128, 1, 1],
            bias: false,
            outputShape: [1, 512, 28, 28],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 512, 28, 28],
            kernelShape: [128, 512, 1, 1],
            bias: false,
            outputShape: [1, 128, 28, 28],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 128, 28, 28],
            kernelShape: [128, 128, 3, 3],
            bias: false,
            outputShape: [1, 128, 28, 28],
            paddings: [1, 1, 1, 1],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 128, 28, 28],
            kernelShape: [512, 128, 1, 1],
            bias: false,
            outputShape: [1, 512, 28, 28],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 512, 28, 28],
            kernelShape: [256, 512, 1, 1],
            bias: false,
            outputShape: [1, 256, 28, 28],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 512, 28, 28],
            kernelShape: [1024, 512, 1, 1],
            bias: false,
            outputShape: [1, 1024, 14, 14],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [2, 2],
            group: 1
        },
        {
            inputShape: [1, 256, 28, 28],
            kernelShape: [256, 256, 3, 3],
            bias: false,
            outputShape: [1, 256, 14, 14],
            paddings: [1, 1, 1, 1],
            dilations: [1, 1],
            strides: [2, 2],
            group: 1
        },
        {
            inputShape: [1, 256, 14, 14],
            kernelShape: [1024, 256, 1, 1],
            bias: false,
            outputShape: [1, 1024, 14, 14],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 1024, 14, 14],
            kernelShape: [256, 1024, 1, 1],
            bias: false,
            outputShape: [1, 256, 14, 14],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 256, 14, 14],
            kernelShape: [256, 256, 3, 3],
            bias: false,
            outputShape: [1, 256, 14, 14],
            paddings: [1, 1, 1, 1],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 256, 14, 14],
            kernelShape: [1024, 256, 1, 1],
            bias: false,
            outputShape: [1, 1024, 14, 14],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 1024, 14, 14],
            kernelShape: [256, 1024, 1, 1],
            bias: false,
            outputShape: [1, 256, 14, 14],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 256, 14, 14],
            kernelShape: [256, 256, 3, 3],
            bias: false,
            outputShape: [1, 256, 14, 14],
            paddings: [1, 1, 1, 1],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 256, 14, 14],
            kernelShape: [1024, 256, 1, 1],
            bias: false,
            outputShape: [1, 1024, 14, 14],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 1024, 14, 14],
            kernelShape: [256, 1024, 1, 1],
            bias: false,
            outputShape: [1, 256, 14, 14],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 256, 14, 14],
            kernelShape: [256, 256, 3, 3],
            bias: false,
            outputShape: [1, 256, 14, 14],
            paddings: [1, 1, 1, 1],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 256, 14, 14],
            kernelShape: [1024, 256, 1, 1],
            bias: false,
            outputShape: [1, 1024, 14, 14],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 1024, 14, 14],
            kernelShape: [256, 1024, 1, 1],
            bias: false,
            outputShape: [1, 256, 14, 14],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 256, 14, 14],
            kernelShape: [256, 256, 3, 3],
            bias: false,
            outputShape: [1, 256, 14, 14],
            paddings: [1, 1, 1, 1],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 256, 14, 14],
            kernelShape: [1024, 256, 1, 1],
            bias: false,
            outputShape: [1, 1024, 14, 14],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 1024, 14, 14],
            kernelShape: [256, 1024, 1, 1],
            bias: false,
            outputShape: [1, 256, 14, 14],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 256, 14, 14],
            kernelShape: [256, 256, 3, 3],
            bias: false,
            outputShape: [1, 256, 14, 14],
            paddings: [1, 1, 1, 1],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 256, 14, 14],
            kernelShape: [1024, 256, 1, 1],
            bias: false,
            outputShape: [1, 1024, 14, 14],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 1024, 14, 14],
            kernelShape: [512, 1024, 1, 1],
            bias: false,
            outputShape: [1, 512, 14, 14],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 1024, 14, 14],
            kernelShape: [2048, 1024, 1, 1],
            bias: false,
            outputShape: [1, 2048, 7, 7],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [2, 2],
            group: 1
        },
        {
            inputShape: [1, 512, 14, 14],
            kernelShape: [512, 512, 3, 3],
            bias: false,
            outputShape: [1, 512, 7, 7],
            paddings: [1, 1, 1, 1],
            dilations: [1, 1],
            strides: [2, 2],
            group: 1
        },
        {
            inputShape: [1, 512, 7, 7],
            kernelShape: [2048, 512, 1, 1],
            bias: false,
            outputShape: [1, 2048, 7, 7],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 2048, 7, 7],
            kernelShape: [512, 2048, 1, 1],
            bias: false,
            outputShape: [1, 512, 7, 7],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 512, 7, 7],
            kernelShape: [512, 512, 3, 3],
            bias: false,
            outputShape: [1, 512, 7, 7],
            paddings: [1, 1, 1, 1],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 512, 7, 7],
            kernelShape: [2048, 512, 1, 1],
            bias: false,
            outputShape: [1, 2048, 7, 7],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 2048, 7, 7],
            kernelShape: [512, 2048, 1, 1],
            bias: false,
            outputShape: [1, 512, 7, 7],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 512, 7, 7],
            kernelShape: [512, 512, 3, 3],
            bias: false,
            outputShape: [1, 512, 7, 7],
            paddings: [1, 1, 1, 1],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
        {
            inputShape: [1, 512, 7, 7],
            kernelShape: [2048, 512, 1, 1],
            bias: false,
            outputShape: [1, 2048, 7, 7],
            paddings: [0, 0, 0, 0],
            dilations: [1, 1],
            strides: [1, 1],
            group: 1
        },
    ];
  }
  
async function testMe(convFunc, validate, count) {
    for(let k = 0; k < testDataSet.length; ++k) {
        const testData = testDataSet[k];
        console.log(`Testing ${JSON.stringify(testData)}`);
        const inputData = createRandomArray(testData.inputShape.reduce((a, b) => a * b));
        const kernelData = createRandomArray(testData.kernelShape.reduce((a, b) => a * b));
        for(let i =0; i < count; ++i) {
            const actual = await convFunc(inputData, testData.inputShape,
                kernelData, testData.kernelShape, undefined, '', 
                testData.dilations, testData.group, testData.paddings, testData.strides);
            if(validate) {
                const epsilon = 0.001;
                const expected = cpuConv(
                    inputData, testData.inputShape, kernelData, testData.kernelShape, 
                    null, '', testData.dilations, testData.group,
                    testData.paddings, testData.strides);
                if(!compareOutputs(actual, expected, epsilon)) {
                    console.error('Expected and Actual did not match');
                    console.log(actual);
                    console.log(expected);
                } else {
                    console.info('Actual and expected matched!');
                }
            }
        }
    }
}
