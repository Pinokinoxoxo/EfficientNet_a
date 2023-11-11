/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.imageclassification.fragments

import android.annotation.SuppressLint
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.CountDownTimer
import android.os.Handler
import android.os.Looper
import android.util.DisplayMetrics
import android.util.Log
import android.view.*
import android.widget.AdapterView
import android.widget.Toast
import androidx.camera.core.AspectRatio
import androidx.camera.core.ImageProxy
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.navigation.Navigation
import androidx.recyclerview.widget.LinearLayoutManager
import org.tensorflow.lite.examples.imageclassification.ImageClassifierHelper
import org.tensorflow.lite.examples.imageclassification.R
import org.tensorflow.lite.examples.imageclassification.databinding.FragmentCameraBinding
import org.tensorflow.lite.task.vision.classifier.Classifications
import java.io.IOException
import java.util.Timer
import java.util.TimerTask
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraFragment : Fragment(), ImageClassifierHelper.ClassifierListener {

    companion object {
        private const val TAG = "Image Classifier"
    }

    private var _fragmentCameraBinding: FragmentCameraBinding? = null
    private val fragmentCameraBinding get() = _fragmentCameraBinding!!
    private var totalClassified = 0
    private var successfulClassified = 0

    private lateinit var imageClassifierHelper: ImageClassifierHelper
    private val classificationResultsAdapter by lazy {
        ClassificationResultsAdapter().apply {
            updateAdapterSize(imageClassifierHelper.maxResults)
        }
    }

    private var currentImageNumber = 1

    override fun onResume() {
        super.onResume()

        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(requireActivity(), R.id.fragment_container)
                .navigate(CameraFragmentDirections.actionCameraToPermissions())
        }
    }

    override fun onDestroyView() {
        _fragmentCameraBinding = null
        super.onDestroyView()
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding = FragmentCameraBinding.inflate(inflater, container, false)
        return fragmentCameraBinding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        imageClassifierHelper =
            ImageClassifierHelper(context = requireContext(), imageClassifierListener = this)

        with(fragmentCameraBinding.recyclerviewResults) {
            layoutManager = LinearLayoutManager(requireContext())
            adapter = classificationResultsAdapter
        }

        // Begin the classification loop
        classifySavedImage()
    }

    private fun loadImageFromAssets(imageName: String): Bitmap? {
        try {
            val inputStream = requireContext().assets.open("Pic/$imageName")
            return BitmapFactory.decodeStream(inputStream)
        } catch (e: IOException) {
            Log.e(TAG, "Error loading image from assets", e)
        }
        return null
    }

    private fun classifySavedImage() {
        val imageName = String.format("test_image_%d.jpg", currentImageNumber)
        val bitmap = loadImageFromAssets(imageName)
        if (bitmap != null) {
            imageClassifierHelper.classify(bitmap, getScreenOrientation())
            // Increment and loop if necessary
            currentImageNumber++
            if (currentImageNumber > 5000) {
                currentImageNumber = 1
            }
        } else {
            Log.e(TAG, "Failed to load image: $imageName")
        }
    }

    private fun getScreenOrientation(): Int {
        val outMetrics = DisplayMetrics()
        val display = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            requireActivity().display
        } else {
            @Suppress("DEPRECATION")
            requireActivity().windowManager.defaultDisplay
        }
        display?.getRealMetrics(outMetrics)
        return display?.rotation ?: 0
    }

    @SuppressLint("NotifyDataSetChanged")
    override fun onError(error: String) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
            classificationResultsAdapter.updateResults(null)
            classificationResultsAdapter.notifyDataSetChanged()
        }
    }

    @SuppressLint("NotifyDataSetChanged")
    override fun onResults(
        results: List<Classifications>?,
        inferenceTime: Long
    ) {
        activity?.runOnUiThread {
            // Show results on bottom sheet
            classificationResultsAdapter.updateResults(results)
            classificationResultsAdapter.notifyDataSetChanged()
            fragmentCameraBinding.bottomSheetLayout.inferenceTimeVal.text =
                String.format("%d ms", inferenceTime)

            // Update the counters
            totalClassified++
            var highestProbLabel = ""
            if (results != null && results.isNotEmpty()) {
                successfulClassified++

                // Find the highest probability label
                results[0].categories.maxByOrNull { it.score }?.let { highestProbCategory ->
                    highestProbLabel = highestProbCategory.label
                }
            }

            // Log the details with the highest probability label
            Log.d(
                TAG,
                "Total Classified: $totalClassified, Successfully Classified: $successfulClassified, Highest Probability Label: $highestProbLabel, Inference Time: ${inferenceTime}ms"
            )

            // After showing results, classify the next image
            classifySavedImage()
        }
    }
}
