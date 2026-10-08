document.addEventListener("DOMContentLoaded", function () {

    /* =====================================================
       NAVIGATION
    ====================================================== */

    const navLinks = document.querySelectorAll(".nav-link");

    navLinks.forEach(function (link) {

        link.addEventListener("click", function () {

            navLinks.forEach(function (item) {
                item.classList.remove("active");
            });

            link.classList.add("active");

        });

    });


    /* =====================================================
       SMOOTH SCROLL
    ====================================================== */

    document
        .querySelectorAll('a[href^="#"]')
        .forEach(function (link) {

            link.addEventListener("click", function (event) {

                const targetId =
                    link.getAttribute("href");

                const target =
                    document.querySelector(targetId);

                if (target) {

                    event.preventDefault();

                    target.scrollIntoView({
                        behavior: "smooth",
                        block: "start"
                    });

                }

            });

        });


    /* =====================================================
       READ FLASK RESULT DATA
    ====================================================== */

    const dataElement =
        document.getElementById("cluster-data");

    let results = null;

    if (dataElement) {

        try {

            const rawData =
                dataElement.textContent.trim();

            if (rawData && rawData !== "null") {

                results = JSON.parse(rawData);

            }

        } catch (error) {

            console.error(
                "Could not read clustering results:",
                error
            );

        }

    }


    /* =====================================================
       FEATURE SELECTION VALIDATION
    ====================================================== */

    const clusteringForm =
        document.getElementById(
            "clustering-form"
        );


    if (clusteringForm) {

        clusteringForm.addEventListener(
            "submit",
            function (event) {

                const checkedFeatures =
                    clusteringForm.querySelectorAll(
                        'input[name="features"]:checked'
                    );


                if (checkedFeatures.length < 2) {

                    event.preventDefault();

                    alert(
                        "Please select at least two customer attributes."
                    );

                    return;

                }


                /* -----------------------------------------
                   Prevent invalid K
                ----------------------------------------- */

                const kElement =
                    document.getElementById("k");

                if (kElement) {

                    const k =
                        parseInt(
                            kElement.value,
                            10
                        );

                    if (
                        Number.isNaN(k) ||
                        k < 2 ||
                        k > 10
                    ) {

                        event.preventDefault();

                        alert(
                            "Number of clusters must be between 2 and 10."
                        );

                    }

                }

            }
        );

    }


    /* =====================================================
       STOP IF THERE ARE NO RESULTS
    ====================================================== */

    if (!results) {

        console.log(
            "No clustering results available."
        );

        return;

    }


    console.log(
        "Clustering results:",
        results
    );


    /* =====================================================
       PLOTLY AVAILABILITY CHECK
    ====================================================== */

    if (typeof Plotly === "undefined") {

        console.error(
            "Plotly could not be loaded."
        );

        return;

    }


    /* =====================================================
       COMMON PLOT SETTINGS
    ====================================================== */

    const commonLayout = {

        template: "plotly_white",

        paper_bgcolor:
            "rgba(0,0,0,0)",

        plot_bgcolor:
            "rgba(0,0,0,0)",

        font: {
            family:
                "Arial, Helvetica, sans-serif",

            color: "#514941"
        },

        hoverlabel: {
            bgcolor: "white",

            bordercolor: "#eadfd6",

            font: {
                color: "#332c27"
            }
        }

    };


    const commonConfig = {

        responsive: true,

        displayModeBar: false,

        scrollZoom: false

    };


    /* =====================================================
       ELBOW CHART
    ====================================================== */

    const elbowElement =
        document.getElementById(
            "elbow-chart"
        );


    if (
        elbowElement &&
        Array.isArray(results.elbow_x) &&
        Array.isArray(results.elbow_y) &&
        results.elbow_x.length > 0 &&
        results.elbow_y.length > 0
    ) {

        const elbowTrace = {

            x: results.elbow_x,

            y: results.elbow_y,

            type: "scatter",

            mode: "lines+markers",

            name: "WCSS",

            marker: {
                size: 8
            },

            line: {
                width: 3
            },

            hovertemplate:
                "K = %{x}<br>" +
                "WCSS = %{y:.2f}" +
                "<extra></extra>"

        };


        const elbowLayout = {

            ...commonLayout,

            title: {
                text:
                    "Elbow Method"
            },

            xaxis: {

                title:
                    "Number of Clusters (K)",

                dtick: 1,

                gridcolor: "#eee5de"

            },

            yaxis: {

                title:
                    "Within-Cluster Sum of Squares",

                gridcolor: "#eee5de"

            },

            margin: {

                t: 65,

                r: 25,

                b: 65,

                l: 85

            }

        };


        Plotly.newPlot(

            elbowElement,

            [elbowTrace],

            elbowLayout,

            commonConfig

        ).catch(function (error) {

            console.error(
                "Elbow chart error:",
                error
            );

        });

    }


    /* =====================================================
       CLUSTER SIZE CHART
    ====================================================== */

    const clusterSizeElement =
        document.getElementById(
            "cluster-size-chart"
        );


    if (
        clusterSizeElement &&
        results.cluster_counts &&
        typeof results.cluster_counts === "object"
    ) {

        const clusters =
            Object.keys(
                results.cluster_counts
            );


        const labels =
            clusters.map(
                function (cluster) {

                    return "Cluster " + cluster;

                }
            );


        const values =
            clusters.map(
                function (cluster) {

                    return Number(
                        results.cluster_counts[
                            cluster
                        ]
                    );

                }
            );


        if (values.length > 0) {

            const sizeTrace = {

                x: labels,

                y: values,

                type: "bar",

                name: "Customers",

                hovertemplate:
                    "%{x}<br>" +
                    "Customers = %{y}" +
                    "<extra></extra>"

            };


            const sizeLayout = {

                ...commonLayout,

                title: {

                    text:
                        "Customers in Each Cluster"

                },

                xaxis: {

                    title:
                        "Customer Cluster",

                    gridcolor: "#eee5de"

                },

                yaxis: {

                    title:
                        "Number of Customers",

                    gridcolor: "#eee5de"

                },

                margin: {

                    t: 65,

                    r: 25,

                    b: 75,

                    l: 75

                }

            };


            Plotly.newPlot(

                clusterSizeElement,

                [sizeTrace],

                sizeLayout,

                commonConfig

            ).catch(function (error) {

                console.error(
                    "Cluster size chart error:",
                    error
                );

            });

        }

    }


    /* =====================================================
       PCA CLUSTER VISUALIZATION
    ====================================================== */

    const clusterElement =
        document.getElementById(
            "cluster-chart"
        );


    if (
        clusterElement &&
        Array.isArray(results.pca_points) &&
        results.pca_points.length > 0
    ) {

        const uniqueClusters =
            [
                ...new Set(
                    results.pca_points.map(
                        function (point) {

                            return point.cluster;

                        }
                    )
                )
            ].sort(
                function (a, b) {

                    return a - b;

                }
            );


        const traces =
            uniqueClusters.map(
                function (cluster) {

                    const clusterPoints =
                        results.pca_points.filter(
                            function (point) {

                                return (
                                    point.cluster ===
                                    cluster
                                );

                            }
                        );


                    return {

                        x:
                            clusterPoints.map(
                                function (point) {

                                    return point.x;

                                }
                            ),

                        y:
                            clusterPoints.map(
                                function (point) {

                                    return point.y;

                                }
                            ),

                        type: "scatter",

                        mode: "markers",

                        name:
                            "Cluster " +
                            cluster,

                        marker: {

                            size: 9,

                            opacity: 0.78

                        },

                        hovertemplate:

                            "Cluster " +
                            cluster +
                            "<br>" +

                            "PC1 = %{x:.2f}<br>" +

                            "PC2 = %{y:.2f}" +

                            "<extra></extra>"

                    };

                }
            );


        const clusterLayout = {

            ...commonLayout,

            title: {

                text:
                    "Customer Cluster Visualization"

            },

            xaxis: {

                title:
                    "Principal Component 1",

                zeroline: false,

                gridcolor: "#eee5de"

            },

            yaxis: {

                title:
                    "Principal Component 2",

                zeroline: false,

                gridcolor: "#eee5de"

            },

            margin: {

                t: 65,

                r: 25,

                b: 80,

                l: 75

            },

            legend: {

                orientation: "h",

                y: -0.2,

                x: 0

            }

        };


        Plotly.newPlot(

            clusterElement,

            traces,

            clusterLayout,

            commonConfig

        ).catch(function (error) {

            console.error(
                "PCA chart error:",
                error
            );

        });

    }


    /* =====================================================
       RESIZE CHARTS
    ====================================================== */

    window.addEventListener(
        "resize",
        function () {

            const chartIds = [

                "elbow-chart",

                "cluster-size-chart",

                "cluster-chart"

            ];


            chartIds.forEach(
                function (id) {

                    const element =
                        document.getElementById(id);


                    if (
                        element &&
                        element.data
                    ) {

                        Plotly.Plots.resize(
                            element
                        );

                    }

                }
            );

        }
    );


    /* =====================================================
       FEATURE CARD INTERACTION
    ====================================================== */

    const featureCards =
        document.querySelectorAll(
            ".feature-card"
        );


    featureCards.forEach(
        function (card) {

            const checkbox =
                card.querySelector(
                    'input[type="checkbox"]'
                );


            if (!checkbox) {
                return;
            }


            card.addEventListener(
                "click",
                function (event) {

                    /*
                     * Do not toggle twice when the actual
                     * checkbox is clicked.
                     */

                    if (
                        event.target.tagName ===
                        "INPUT"
                    ) {

                        return;

                    }


                    checkbox.checked =
                        !checkbox.checked;


                    checkbox.dispatchEvent(
                        new Event(
                            "change",
                            {
                                bubbles: true
                            }
                        )
                    );

                }
            );

        }
    );


    /* =====================================================
       UPDATE FEATURE COUNTER
    ====================================================== */

    function updateFeatureCounter() {

        const selected =
            document.querySelectorAll(
                'input[name="features"]:checked'
            );


        const counter =
            document.getElementById(
                "selected-feature-count"
            );


        if (counter) {

            counter.textContent =
                selected.length;

        }

    }


    document
        .querySelectorAll(
            'input[name="features"]'
        )
        .forEach(
            function (checkbox) {

                checkbox.addEventListener(
                    "change",
                    updateFeatureCounter
                );

            }
        );


    updateFeatureCounter();


    /* =====================================================
       ACTIVE NAVIGATION WHILE SCROLLING
    ====================================================== */

    const sections = [

        document.getElementById("dashboard"),

        document.getElementById("dataset"),

        document.getElementById("clustering"),

        document.getElementById("results")

    ].filter(
        function (section) {

            return section !== null;

        }
    );


    if (sections.length > 0) {

        const observer =
            new IntersectionObserver(
                function (entries) {

                    entries.forEach(
                        function (entry) {

                            if (
                                entry.isIntersecting
                            ) {

                                const activeId =
                                    entry.target.id;


                                navLinks.forEach(
                                    function (link) {

                                        link.classList.toggle(

                                            "active",

                                            link.getAttribute(
                                                "href"
                                            ) ===
                                            "#" +
                                            activeId

                                        );

                                    }
                                );

                            }

                        }
                    );

                },
                {
                    threshold: 0.25
                }
            );


        sections.forEach(
            function (section) {

                observer.observe(
                    section
                );

            }
        );

    }

});