package com.example.omu.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Voice Activity Detection (VAD) Engine using Silero VAD via ONNX Runtime.
 * Optimized for 16kHz mono PCM audio in 32ms (512 samples) frames.
 */
class VadEngine(context: Context) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private var h = FloatArray(128)
    private var c = FloatArray(128)

    init {
        val modelBytes = context.assets.open("silero_vad.onnx").use { it.readBytes() }
        val options = OrtSession.SessionOptions()
        session = env.createSession(modelBytes, options)
    }
    
    fun isSpeech(audioFrame: FloatArray): Float {
        if (audioFrame.size != 512) return 0f

        val inputs = mutableMapOf<String, OnnxTensor>()
        val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(audioFrame), longArrayOf(1, 512))
        
        val srBuffer = LongBuffer.allocate(1)
        srBuffer.put(16000L)
        srBuffer.rewind()
        val srTensor = OnnxTensor.createTensor(env, srBuffer, longArrayOf(1))

        val hTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(h), longArrayOf(2, 1, 64))
        val cTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(c), longArrayOf(2, 1, 64))

        inputs["input"] = inputTensor
        inputs["sr"] = srTensor
        inputs["h"] = hTensor
        inputs["c"] = cTensor

        return try {
            val results = session.run(inputs)

            @Suppress("UNCHECKED_CAST")
            val output = results[0].value as Array<FloatArray>
            val probability = output[0][0]

            @Suppress("UNCHECKED_CAST")
            val hOut = results[1].value as Array<Array<FloatArray>>
            @Suppress("UNCHECKED_CAST")
            val cOut = results[2].value as Array<Array<FloatArray>>
            
            h = flattenRNNState(hOut)
            c = flattenRNNState(cOut)

            probability
        } catch (e: Exception) {
            android.util.Log.e("VadEngine", "VAD Inference failed", e)
            0f
        } finally {
            inputTensor.close()
            srTensor.close()
            hTensor.close()
            cTensor.close()
        }
    }

    private fun flattenRNNState(state: Array<Array<FloatArray>>): FloatArray {
        val flat = FloatArray(128)
        var idx = 0
        for (i in 0 until 2) {
            for (j in 0 until 1) {
                for (k in 0 until 64) {
                    flat[idx++] = state[i][j][k]
                }
            }
        }
        return flat
    }
    
    fun reset() {
        h = FloatArray(128)
        c = FloatArray(128)
    }

    fun close() {
        session.close()
        env.close()
    }
}
