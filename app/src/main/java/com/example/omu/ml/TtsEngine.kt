package com.example.omu.ml

import ai.onnxruntime.*

class TtsEngine(private val modelPath: String) {
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    fun initialize() {
        ortEnv = OrtEnvironment.getEnvironment()


        val sessionOptions = OrtSession.SessionOptions().apply {
            // Enable XNNPACK for optimized CPU performance on Android
            addXnnpack(mapOf("intra_op_num_threads" to "4"))
        }

        ortSession = ortEnv?.createSession(modelPath, sessionOptions)
    }

    fun close() {
        ortSession?.close()
        ortEnv?.close()
    }
}


