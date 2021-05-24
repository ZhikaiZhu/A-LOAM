
#ifndef INCLUDE_MAP_BUFFER_HPP_
#define INCLUDE_MAP_BUFFER_HPP_

#include <iostream>
#include <map>

template <typename Meas>
class MapBuffer {
    public:
        std::map<double, Meas> measMap_;
        typename std::map<double, Meas>::iterator itMeas_;
        int size;

        MapBuffer() = default;
        virtual ~MapBuffer() {}

        bool allocate(const int sizeBuffer) {
            if (sizeBuffer <= 0) {
                return false;
            }
            else {
                size = sizeBuffer;
                return true;
            }
        }

        void clear() {
            measMap_.clear();
        }

        void clean(double t) {
            while (measMap_.size() >= 1 && measMap_.begin()->first <= t) {
                measMap_.erase(measMap_.begin());
            }
        }

        int getSize() {
            return measMap_.size();
        }

        bool getNextTime(double curTime, double &nextTime) {
            itMeas_ = measMap_.upper_bound(curTime);
            if (itMeas_ != measMap_.end()) {
                nextTime = itMeas_->first;
                return true;
            }
            else {
                return false;
            }
        }

        bool getFirstTime(double &firstTime) {
            if (!measMap_.empty()) {
                firstTime = measMap_.begin()->first;
                return true;
            }
            else {
                return false;
            }
        }

        bool getLastTime(double &lastTime) {
            if (!measMap_.empty()) {
                lastTime = measMap_.rbegin()->first;
                return true;
            }
            else {
                return false;
            }
        }

        bool getFirstMeas(Meas &firstMeas) {
            if (!measMap_.empty()) {
                firstMeas = measMap_.begin()->second;
                return true;
            }
            else {
                return false;
            }
        }

        bool getLastMeas(Meas &lastMeas) {
            if (!measMap_.empty()) {
                lastMeas = measMap_.rbegin()->second;
                return true;
            }
            else {
                return false;
            }
        }

        bool empty() {
            return measMap_.empty();
        }

        bool hasMeasurement(double t) {
            return measMap_.count(t) > 0;
        }

        void addMeas(const Meas& meas, const double &t) {
            measMap_.insert(std::make_pair(t, meas));
            if (measMap_.size() > size) {
                measMap_.erase(measMap_.begin());
            }
        }

        void printContainer() {
            itMeas_ = measMap_.begin();
            while (measMap_.size() >= 1 && itMeas_ != measMap_.end()) {
                std::cout << itMeas_->second << " ";
                itMeas_++;
            }
        }
};

#endif // INCLUDE_MAP_BUFFER_HPP_