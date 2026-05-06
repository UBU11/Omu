package com.example.omu.audio

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder


class MicRecorder {
    @androidx.annotation.RequiresPermission(android.Manifest.permission.RECORD_AUDIO)
    fun start(): Flow<ByteArray> = channelFlow {
        val sampleRate = 16000
        val channelConfig = AudioFormat.CHANNEL_IN_MONO
        val audioFormat = AudioFormat.ENCODING_PCM_16BIT
        val minBufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
        
        if (minBufferSize == AudioRecord.ERROR_BAD_VALUE || minBufferSize == AudioRecord.ERROR) {
            return@channelFlow
        }

        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            channelConfig,
            audioFormat,
            minBufferSize
        )

        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            recorder.release()
            return@channelFlow
        }

        recorder.startRecording()

        val frameSize = (sampleRate * 0.032).toInt() * 2 // 32ms of 16-bit audio
        val buffer = ByteArray(frameSize)

        try {
            withContext(Dispatchers.IO) {
                while (true) {
                    val bytesRead = recorder.read(buffer, 0, frameSize)
                    if (bytesRead > 0) {
                        val frame = buffer.copyOf(bytesRead)
                        send(frame)
                    } else if (bytesRead < 0) {
                        break
                    }
                }
            }
        } finally {
            recorder.stop()
            recorder.release()
        }
    }
}