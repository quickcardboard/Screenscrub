plugins {
    alias(libs.plugins.kotlin.jvm)
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}
kotlin {
    compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) }
}

dependencies {
    // OpenCV Java API. On Android the classes come from the OpenCV AAR (same 4.9 API);
    // on the desktop they come from openpnp, which bundles native libraries.
    compileOnly(libs.opencv.jvm)
    testImplementation(libs.opencv.jvm)
    testImplementation(libs.junit)
}

// Tuning harness: ./gradlew :vision:tune -Psamples=/path/to/samples
tasks.register<JavaExec>("tune") {
    group = "verification"
    description = "Classify samples/{screen,document,other} and print a confusion matrix."
    dependsOn("testClasses")
    classpath = sourceSets["test"].runtimeClasspath
    mainClass.set("com.screenscrub.vision.TuningHarnessKt")
    val dir = (project.findProperty("samples") as String?) ?: rootProject.file("samples").absolutePath
    args(dir)
}
